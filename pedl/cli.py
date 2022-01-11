#!/usr/bin/env python

import argparse
import sys
from collections import defaultdict
from pathlib import Path
import random

from tqdm import tqdm

from pedl.predict import predict
from pedl.utils import build_summary_table, \
    Entity
from pedl.data_getter import DataGetterAPI
from pedl import pubtator_elasticsearch

random.seed(42)


def summarize(args):
    if not args.out:
        file_out = (args.path_to_files.parent / args.path_to_files.name).with_suffix(".tsv")
    else:
        file_out = args.out
    with open(file_out, "w") as f:
        f.write(f"p1\tassociation type\tp2\tscore (sum)\tscore (max)\tpmids\n")
        for row in build_summary_table(args.path_to_files, score_cutoff=args.cutoff,
                                       no_association_type=args.no_association_type):
            f.write(f"{row[0]}\t{row[1]}\t{row[2]}\t{row[3]:.2f}\t{row[4]:.2f}{row[5]}\n")


def build_training_set(args):
    pair_to_relations = defaultdict(set)
    pmid_to_pairs = defaultdict(set)

    gene_universe = set()
    chemical_universe = set()

    with args.triples.open() as f:
        for line in f:
            fields = line.strip().split("\t")
            if fields:
                type_head, cuid_head, type_tail, cuid_tail, rel = fields
                head = Entity(cuid=cuid_head, type=type_head)
                tail = Entity(cuid=cuid_tail, type=type_tail)
                pair_to_relations[(head, tail)].add(rel)

                if head.type == "Chemical":
                    chemical_universe.add(head.cuid)
                elif head.type == "Gene":
                    gene_universe.add(head.cuid)
                else:
                    raise ValueError(head.type)
                if tail.type == "Chemical":
                    chemical_universe.add(tail.cuid)
                elif tail.type == "Gene":
                    gene_universe.add(tail.cuid)
                else:
                    raise ValueError(tail.type)

    data_getter = DataGetterAPI(chemical_universe=chemical_universe,
                                gene_universe=gene_universe,
                                expand_species=args.expand_species)

    all_pmids = set()

    for pair, relations in tqdm(list(pair_to_relations.items()),
                                desc="Preparing Crawl"):
        head, tail = pair[:2]
        shared_pmids = data_getter.get_pmids(head) & data_getter.get_pmids(tail)

        all_pmids.update(shared_pmids)
        for pmid in shared_pmids:
            pmid_to_pairs[pmid].add((head, tail))

    with open(str(args.out) + "." + str(args.worker_id), "w") as f, open(str(args.out_blinded) + "." + str(args.worker_id), "w") as f_blinded:
        for i, pmid in enumerate(tqdm(sorted(pmid_to_pairs), desc="Crawling")):
            if not (i % args.n_worker) == args.worker_id:
                continue
            pairs = pmid_to_pairs[pmid]
            docs = list(data_getter.get_documents([pmid]))[0]
            if docs:
                doc = docs[0]
                for head, tail in pairs:
                    sentences = data_getter.get_sentences_from_document(entity1=head,
                                                                        entity2=tail,
                                                                        document=doc)
                    relations = pair_to_relations[(head, tail)]
                    for sentence in sentences:
                        f.write("\t".join([head.type, head.cuid, tail.type, tail.cuid, ",".join(relations), sentence.text, sentence.pmid]) + "\n")
                        f_blinded.write("\t".join([head.type, head.cuid, tail.type, tail.cuid, ",".join(relations), sentence.text_blinded, sentence.pmid]) + "\n")


def rebuild_pubtator_index(args):
    really_continue = input("This will delete the pubtator index and rebuild it. Do you want to continue? Type 'yes':\n")
    if really_continue != "yes":
        sys.exit(0)
    pubtator_elasticsearch.build_index(pubtator_path=args.pubtator,
                                       n_processes=args.n_processes)




def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_predict = subparsers.add_parser("predict")

    parser_predict.add_argument('--p1', required=True, nargs="+")
    parser_predict.add_argument('--p2', required=True, nargs="+")
    parser_predict.add_argument('--out', type=Path, required=True)
    parser_predict.add_argument('--model', default="leonweber/PEDL")
    parser_predict.add_argument('--batch_size', default=None, type=int)
    parser_predict.add_argument('--pubtator')
    parser_predict.add_argument('--device', default=None)
    parser_predict.add_argument('--topk', type=int, default=None)
    parser_predict.add_argument('--cutoff', type=float, default=0.01)
    parser_predict.add_argument('--max_bag_size', type=int, default=1000)
    parser_predict.add_argument('--api_fallback', action="store_true")
    parser_predict.add_argument('--skip_reverse', action="store_true")
    parser_predict.add_argument('--skip_invalid', action="store_true")
    parser_predict.add_argument('--verbose', action="store_true")
    parser_predict.add_argument('--expand_species', nargs="*")
    parser_predict.add_argument('--multi_sentence', action="store_true")
    parser_predict.add_argument('--num_workers', type=int, default=1)
    parser_predict.add_argument('--worker_id', type=int, default=0)
    parser_predict.add_argument('--pmids', type=Path, default=None)
    parser_predict.set_defaults(func=predict)

    parser_summarize = subparsers.add_parser("summarize")
    parser_summarize.set_defaults(func=summarize)

    parser_summarize.add_argument("path_to_files", type=Path)
    parser_summarize.add_argument("--out", type=Path, default=None)
    parser_summarize.add_argument("--cutoff", type=float, default=0.0)
    parser_summarize.add_argument('--no_association_type', action="store_true")


    ## Build Training Data
    parser_build_training_set = subparsers.add_parser("build_training_set")
    parser_build_training_set.set_defaults(func=build_training_set)

    parser_build_training_set.add_argument("--out", type=Path, required=True)
    parser_build_training_set.add_argument("--out_blinded", type=Path, required=True)
    parser_build_training_set.add_argument('--expand_species', nargs="*")
    parser_build_training_set.add_argument("--n_worker", default=1, type=int)
    parser_build_training_set.add_argument("--worker_id", default=0, type=int)
    parser_build_training_set.add_argument("--triples", type=Path, required=True)
    parser_build_training_set.add_argument('--pubtator', action="store_true")


    ## Rebuild PubTator
    parser_rebuild_pubtator_index = subparsers.add_parser("rebuild_pubtator_index")
    parser_rebuild_pubtator_index.set_defaults(func=rebuild_pubtator_index)
    parser_rebuild_pubtator_index.add_argument("--pubtator", required=True, type=Path)
    parser_rebuild_pubtator_index.add_argument("--n_processes", type=int, default=None)

    args = parser.parse_args()

    args.func(args)

if __name__ == '__main__':
    main()
