import logging
import os
import sys
import uuid
import hydra
from omegaconf import DictConfig
from pathlib import Path
from typing import Tuple, Set

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from pedl.data_getter import DataGetterPubtator, DataGetterAPI
from pedl.dataset import PEDLDataset
from pedl.model import BertForDistantSupervision
from pedl.utils import get_hgnc_symbol_to_gene_id, get_geneid_to_name, Entity


PREFIX_PROCESSED_PAIRS = ".pairs_processed"


def get_processed_pairs(dir_out: Path) -> Set[Tuple[str, str]]:
    processed_pairs = set()
    for file in dir_out.glob(PREFIX_PROCESSED_PAIRS + "*"):
        with file.open() as f:
            for line in f.read().split("\n"):
                processed_pairs.add(tuple(line.split("\t")))

    return processed_pairs

@torch.no_grad()
@hydra.main(config_path="./configs/predict", config_name="default.yaml")
def predict(config: DictConfig):

    hgnc_to_gene_id = get_hgnc_symbol_to_gene_id()

    if config.verbose:
        logging.basicConfig(level=logging.INFO)

    if len(config.p1) == 1 and os.path.exists(config.p1[0]):
        with open(config.p1[0]) as f:
            p1s = f.read().strip().split("\n")
    elif len(config.p1) == 1 and config.p1[0] == "all_human_genes":
        p1s = sorted(hgnc_to_gene_id.keys())
    else:
        p1s = config.p1

    if len(config.p2) == 1 and os.path.exists(config.p2[0]):
        with open(config.p2[0]) as f:
            p2s = f.read().strip().split("\n")
    elif len(config.p2) == 1 and config.p2[0] == "all_human_genes":
        p2s = sorted(hgnc_to_gene_id.keys())
    else:
        p2s = config.p2

    maybe_mapped_p1s = []
    for p1 in p1s:
        if not p1.isnumeric():
            if not config.skip_invalid:
                assert p1 in hgnc_to_gene_id, f"{p1} is neither a valid HGNC symbol nor a Entrez gene id"
            elif p1 not in hgnc_to_gene_id:
                continue
            maybe_mapped_p1s.append(hgnc_to_gene_id[p1])
        else:
            maybe_mapped_p1s.append(p1)

    maybe_mapped_p2s = []
    for p2 in p2s:
        if not p2.isnumeric():
            if not config.skip_invalid:
                assert p2 in hgnc_to_gene_id, f"{p2} is neither a valid HGNC symbol nor a Entrez gene id"
            elif p2 not in hgnc_to_gene_id:
                continue
            maybe_mapped_p2s.append(hgnc_to_gene_id[p2])
        else:
            maybe_mapped_p2s.append(p2)

    heads = [Entity(cuid, "Gene") for cuid in maybe_mapped_p1s]
    tails = [Entity(cuid, "Gene") for cuid in maybe_mapped_p2s]

    if config.num_workers > 1:
        heads = sorted(heads)
        heads = [head for i, head in enumerate(heads) if i % config.num_workers == config.worker_id]

    processed_pairs = get_processed_pairs(Path(config.out))

    geneid_to_name = get_geneid_to_name()
    if len(heads) * len(tails) > 100 and not config.pubtator:
        print(f"Using PEDL without a local PubTator copy is only supported for small queries up to 100 protein pairs. Your query contains {len(pairs_to_query)} pairs. Aborting.")
        sys.exit(1)

    if not config.device:
        if torch.cuda.is_available():
            config.device = "cuda"
        else:
            config.device = "cpu"

    universe = set(maybe_mapped_p1s + maybe_mapped_p2s)

    if config.pubtator:
        data_getter = DataGetterPubtator(address=config.pubtator)
    else:
        data_getter = DataGetterAPI(gene_universe=universe,
                                    expand_species=config.expand_species,
                                    blind_entity_types={"Gene"}
                                    )

    dataset = PEDLDataset(heads=heads,
                          tails=tails,
                          skip_pairs=processed_pairs,
                          base_model="leonweber/PEDL",
                          data_getter=data_getter,
                          sentence_max_length=500,
                          max_bag_size=config.max_bag_size)

    model = BertForDistantSupervision.from_pretrained(config.model,
                                                      tokenizer=dataset.tokenizer)
    if "cuda" in config.device:
        model.bert = nn.DataParallel(model.bert)
    model.eval()
    model.to(config.device)

    model.config.e1_id = dataset.tokenizer.convert_tokens_to_ids("<e1>")
    model.config.e2_id = dataset.tokenizer.convert_tokens_to_ids("<e2>")


    os.makedirs(config.out, exist_ok=True)

    dataloader = DataLoader(dataset, num_workers=4, batch_size=1,
                            collate_fn=model.collate_fn, prefetch_factor=100)
    with (Path(config.out) / f"{PREFIX_PROCESSED_PAIRS}_{uuid.uuid4()}").open("w") as f_pairs_processed:
        for datapoint in tqdm(dataloader):
            head, tail = datapoint["pair"]
            if "sentences" not in datapoint:
                f_pairs_processed.write(f"{head}\t{tail}\n")
                continue

            name1 = geneid_to_name.get(head.cuid, head.cuid)
            name2 = geneid_to_name.get(tail.cuid, tail.cuid)

            file_out = Path(config.out) / f"{name1}_{name2}.txt"

            if head == tail:
                f_pairs_processed.write(f"{head}\t{tail}\n")
                continue

            if "cuda" in config.device:
                with torch.cuda.amp.autocast():
                    x, meta = model.forward_batched(**datapoint["encoding"],
                                                    batch_size=config.batch_size)
                    probs = torch.sigmoid(meta["alphas_by_rel"])
            else:
                x, meta = model.forward_batched(**datapoint["encoding"],
                                                batch_size=config.batch_size)
                probs = torch.sigmoid(meta["alphas_by_rel"])

            if (probs < config.cutoff).all():
                f_pairs_processed.write(f"{head}\t{tail}\n")
                continue

            with file_out.open("w") as f:
                for max_score in torch.sort(probs.view(-1), descending=True)[0]:
                    if max_score.item() < config.cutoff:
                        continue
                    for i, j in zip(*torch.where(probs == max_score)):
                        label = PEDLDataset.id_to_label[j.item()]
                        sentence = datapoint["sentences"][i]
                        f.write(f"{label}\t{max_score.item():.2f}\t{sentence.pmid}\t{sentence.text}\tPEDL\n\n")
            f_pairs_processed.write(f"{head}\t{tail}\n")


if __name__ == '__main__':
    predict()