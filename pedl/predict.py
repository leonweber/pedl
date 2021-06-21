#!/usr/bin/env python

import argparse
import itertools
import logging
import os
import sys
from pathlib import Path
from torch import nn
from tqdm import tqdm
import torch
from transformers import BertTokenizerFast

from pedl.model import BertForDistantSupervision
from pedl.dataset import PEDLDataset
from pedl.utils import DataGetter, get_geneid_to_name, chunks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--p1', required=True, nargs="+")
    parser.add_argument('--p2', required=True, nargs="+")
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--model', default="leonweber/PEDL")
    parser.add_argument('--dbs', nargs="*", choices=["pid", "reactome", "netpath",
                                                     "kegg", "panther", "humancyc"])
    parser.add_argument('--pubtator', type=Path)
    parser.add_argument('--device', default=None)
    parser.add_argument('--topk', type=int, default=None)
    parser.add_argument('--cutoff', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--api_fallback', action="store_true")
    parser.add_argument('--verbose', action="store_true")
    parser.add_argument('--expand_species', nargs="*")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    if len(args.p1) == 1 and os.path.exists(args.p1[0]):
        with open(args.p1[0]) as f:
            p1s = f.read().strip().split("\n")
    else:
        p1s = args.p1

    if len(args.p2) == 1 and os.path.exists(args.p2[0]):
        with open(args.p2[0]) as f:
            p2s = f.read().strip().split("\n")
    else:
        p2s = args.p2

    pairs_to_query = []
    for p1 in p1s:
        for p2 in p2s:
            pairs_to_query.append((p1, p2))
    if len(pairs_to_query) > 100 and not args.pubtator:
        print(f"Using PEDL without a local PubTator copy is only supported for small queries up to 100 protein pairs. Your query contains {len(pairs_to_query)} pairs. Aborting.")
        sys.exit(1)

    if not args.device:
        if torch.cuda.is_available():
            args.device = "cuda"
        else:
            args.device = "cpu"

    model = BertForDistantSupervision.from_pretrained(args.model)
    if "cuda" in args.device:
        model.bert = nn.DataParallel(model.bert)
    model.eval()
    model.to(args.device)

    universe = set(p1s + p2s)
    data_getter = DataGetter(universe, local_pubtator=args.pubtator,
                             api_fallback=args.api_fallback,
                             expand_species=args.expand_species
                             )
    tokenizer = BertTokenizerFast.from_pretrained(args.model)
    tokenizer.add_special_tokens({ 'additional_special_tokens': ['<e1>','</e1>', '<e2>', '</e2>'] +
                                                                     [f'<protein{i}/>' for i in range(1, 47)]})
    model.config.e1_id = tokenizer.convert_tokens_to_ids("<e1>")
    model.config.e2_id = tokenizer.convert_tokens_to_ids("<e2>")

    geneid_to_name = get_geneid_to_name()

    os.makedirs(args.out, exist_ok=True)

    if args.dbs:
        try:
            from pedl.database import PathwayCommonsDB
        except ImportError:
            print("Harvesting interactions from databases requires indra."
                  "Please install via `pip install indra' for obtaining additional interactions from databases.")
            sys.exit(1)
        logging.info("Preparing databases")
        dbs = [PathwayCommonsDB(i, gene_universe=universe) for i in args.dbs]
    else:
        dbs = []


    pbar = tqdm(total=len(p1s)*len(p2s)*2)
    for i in range(2):
        if i == 0:
            current_p1s = p1s
            current_p2s = p2s
        else:
            current_p1s = p2s
            current_p2s = p1s

        for p1 in current_p1s:
            for p2 in current_p2s:
                name1 = geneid_to_name.get(p1, p1)
                name2 = geneid_to_name.get(p2, p2)
                pbar.set_description(f"{name1}-{name2}")
                if p1 == p2:
                    pbar.update()
                    continue

                processed_db_results = set()
                path_out = args.out / f"{name1}-{name2}.txt"
                with open(path_out, "w") as f:
                    for db in dbs:
                        stmts = db.get_statements(p1, p2)
                        for stmt in stmts:
                            pmids = ",".join(i.pmid for i in stmt.evidence)
                            label = stmt.__class__.__name__
                            provenances = ",".join(i.source_id for i in stmt.evidence)
                            db_result = f"{label}\t1.0\t{pmids}\t{provenances}\t{db.name}\n\n"
                            if db_result not in processed_db_results:
                                f.write(db_result)
                            processed_db_results.add(db_result)

                    probs = []
                    sentences = sorted(data_getter.get_sentences(p1, p2),
                                       key=lambda x: len(x.text))

                    if len(sentences) > args.batch_size:
                        pbar_pred = tqdm(desc="Predicting", total=len(sentences))
                    else:
                        pbar_pred = None
                    for sentences_batch in list(chunks(sentences, args.batch_size)):
                        tensors = tokenizer.batch_encode_plus([i.text_blinded for i in sentences_batch],
                                                              truncation=True, max_length=512)
                        max_length = max(len(i) for i in tensors["input_ids"])
                        tensors = tokenizer.pad(tensors, max_length=max_length,
                                                return_tensors="pt")
                        input_ids = tensors["input_ids"].to(args.device)
                        attention_mask = tensors["attention_mask"].to(args.device)
                        with torch.no_grad():
                            x, meta = model(input_ids, attention_mask)
                        probs_batch = torch.sigmoid(meta["alphas_by_rel"])
                        probs.append(probs_batch)
                        if pbar_pred:
                            pbar_pred.update(len(sentences_batch))


                    if not sentences:
                        pbar.update()
                        if os.path.getsize(path_out) == 0:
                            os.remove(path_out)
                        continue

                    probs = torch.cat(probs)

                    if (probs < args.cutoff).all():
                        pbar.update()
                        if os.path.getsize(path_out) == 0:
                            os.remove(path_out)
                        continue

                    processed_sentences = set()
                    for max_score in torch.sort(probs.view(-1), descending=True)[0]:
                        if max_score.item() < args.cutoff:
                            continue
                        for i, j in zip(*torch.where(probs == max_score)):
                            label = PEDLDataset.id_to_label[j.item()]
                            sentence = sentences[i]
                            sentence_signature = (sentence.get_unmarked_text(), label)
                            if sentence_signature not in processed_sentences:
                                f.write(f"{label}\t{max_score.item():.2f}\t{sentence.pmid}\t{sentence.text}\tPEDL\n\n")
                            processed_sentences.add(sentence_signature)

                pbar.update()
                if os.path.getsize(path_out) == 0:
                    os.remove(path_out)


if __name__ == '__main__':
    main()
