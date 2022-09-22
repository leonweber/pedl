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
from pedl.datasets.pedl_dataset import PEDLDataset
from pedl.bert_for_distant_supervision import BertForDistantSupervision
from pedl.utils import get_hgnc_symbol_to_gene_id, get_geneid_to_name, Entity, maybe_mapped_entities, \
    get_mesh_id_to_chem_name


PREFIX_PROCESSED_PAIRS = ".pairs_processed"

def get_processed_pairs(dir_out: Path) -> Set[Tuple[str, str]]:
    processed_pairs = set()
    for file in dir_out.glob(PREFIX_PROCESSED_PAIRS + "*"):
        with file.open() as f:
            for line in f.read().split("\n"):
                processed_pairs.add(tuple(line.split("\t")))
    return processed_pairs


@torch.no_grad()
@hydra.main(config_path="./configs", config_name="predict.yaml")
def predict(cfg: DictConfig):
    if "drug" in cfg.type1 or "drug" in cfg.type2:
        model_name = cfg.drug_model
        if "drug" in cfg.type1:
            head_id_to_entity = get_mesh_id_to_chem_name()
            tail_id_to_entity = get_hgnc_symbol_to_gene_id()
            head_type = "Chemical"
            tail_type = "Gene"
        else:
            head_id_to_entity = get_hgnc_symbol_to_gene_id()
            tail_id_to_entity = get_mesh_id_to_chem_name()
            head_type = "Gene"
            tail_type = "Chemical"
    else:
        model_name = cfg.prot_model
        head_id_to_entity = tail_id_to_entity = get_hgnc_symbol_to_gene_id()
        head_type = tail_type = "Gene"

    if cfg.verbose:
        logging.basicConfig(level=logging.INFO)

    e1s = get_entity_list(cfg.e1, head_id_to_entity)
    e2s = get_entity_list(cfg.e2, tail_id_to_entity)
    maybe_mapped_e1s = maybe_mapped_entities(e1s, head_id_to_entity, cfg.skip_invalid)
    maybe_mapped_e2s = maybe_mapped_entities(e2s, tail_id_to_entity, cfg.skip_invalid)

    heads = [Entity(cuid, head_type) for cuid in maybe_mapped_e1s]
    tails = [Entity(cuid, tail_type) for cuid in maybe_mapped_e2s]

    if cfg.num_workers > 1:
        heads = sorted(heads)
        heads = [head for i, head in enumerate(heads) if i % cfg.num_workers == cfg.worker_id]

    processed_pairs = get_processed_pairs(Path(cfg.out))

    geneid_to_name = get_geneid_to_name()
    if len(heads) * len(tails) > 100 and not cfg.pubtator:
        print(f"Using PEDL without a local PubTator copy is only supported for small queries up to 100 protein pairs. Your query contains {len(e1s)} pairs. Aborting.")
        sys.exit(1)

    if not cfg.device:
        if torch.cuda.is_available():
            cfg.device = "cuda"
        else:
            cfg.device = "cpu"

    universe = set(maybe_mapped_e1s + maybe_mapped_e2s)

    if cfg.pubtator:
        data_getter = DataGetterPubtator(address=cfg.pubtator,
                                         entity_marker=cfg.entities.entity_marker.values()
                                         )
    else:
        data_getter = DataGetterAPI(gene_universe=universe,
                                    expand_species=cfg.expand_species,
                                    blind_entity_types={head_type, tail_type},
                                    entity_marker=cfg.entities.entity_marker
                                    )
    dataset = PEDLDataset(heads=heads,
                          tails=tails,
                          skip_pairs=processed_pairs,
                          base_model=model_name,
                          data_getter=data_getter,
                          sentence_max_length=500,
                          max_bag_size=cfg.max_bag_size,
                          entity_marker=cfg.entities.entity_marker,
                          max_length=cfg.max_sequence_length
                          )
    model = BertForDistantSupervision.from_pretrained(model_name,
                                                      tokenizer=dataset.tokenizer,
                                                      local_model=cfg.local_model,
                                                      use_cls=cfg.use_cls,
                                                      use_starts=cfg.use_starts,
                                                      use_ends=cfg.use_ends,
                                                      entity_embeddings=cfg.entity_embeddings)
    if "cuda" in cfg.device:
        model.transformer = nn.DataParallel(model.transformer)
    model.eval()
    model.to(cfg.device)
    model.config.e1_id = dataset.tokenizer.convert_tokens_to_ids(cfg.entities.entity_marker["head_start"])
    model.config.e2_id = dataset.tokenizer.convert_tokens_to_ids(cfg.entities.entity_marker["tail_start"])

    os.makedirs(cfg.out, exist_ok=True)

    dataloader = DataLoader(dataset, num_workers=cfg.num_workers, batch_size=cfg.batch_size,
                            collate_fn=model.collate_fn, prefetch_factor=100)
    with (Path(cfg.out) / f"{PREFIX_PROCESSED_PAIRS}_{uuid.uuid4()}").open("w") as f_pairs_processed:
        for datapoint in tqdm(dataloader):
            head, tail = datapoint["pair"]
            if "sentences" not in datapoint:
                f_pairs_processed.write(f"{head}\t{tail}\n")
                continue

            name1 = geneid_to_name.get(head.cuid, head.cuid)
            name2 = geneid_to_name.get(tail.cuid, tail.cuid)

            file_out = Path(cfg.out) / f"{name1}_{name2}.txt"

            if head == tail:
                f_pairs_processed.write(f"{head}\t{tail}\n")
                continue

            if "cuda" in cfg.device:
                with torch.cuda.amp.autocast():
                    x, meta = model.forward_batched(**datapoint["encoding"],
                                                    batch_size=cfg.batch_size)
                    probs = torch.sigmoid(meta["alphas_by_rel"])
            else:
                x, meta = model.forward_batched(**datapoint["encoding"],
                                                batch_size=cfg.batch_size)
                probs = torch.sigmoid(meta["alphas_by_rel"])

            if (probs < cfg.cutoff).all():
                f_pairs_processed.write(f"{head}\t{tail}\n")
                continue

            with file_out.open("w") as f:
                for max_score in torch.sort(probs.view(-1), descending=True)[0]:
                    if max_score.item() < cfg.cutoff:
                        continue
                    for i, j in zip(*torch.where(probs == max_score)):
                        label = PEDLDataset.id_to_label[j.item()]
                        sentence = datapoint["sentences"][i]
                        f.write(f"{label}\t{max_score.item():.2f}\t{sentence.pmid}\t{sentence.text}\tPEDL\n\n")
            f_pairs_processed.write(f"{head}\t{tail}\n")


def get_entity_list(entity, normalized_entity_ids):
    if isinstance(entity, str):
        entity = entity.split()
    if len(entity) == 1 and os.path.exists(entity[0]):
        with open(entity[0]) as f:
            p1s = f.read().strip().split("\n")
    elif len(entity) == 1 and entity[0] == "all":
        p1s = sorted(normalized_entity_ids.keys())
    else:
        p1s = entity
    return p1s


if __name__ == '__main__':
    predict()
