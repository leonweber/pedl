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

ID_MAPPING = {'Chemical': get_mesh_id_to_chem_name,
              'Gene': get_hgnc_symbol_to_gene_id}


def get_processed_pairs(dir_out: Path) -> Set[Tuple[str, str]]:
    processed_pairs = set()
    for file in dir_out.glob(PREFIX_PROCESSED_PAIRS + "*"):
        with file.open() as f:
            for line in f.read().split("\n"):
                processed_pairs.add(tuple(line.split("\t")))
    return processed_pairs


@torch.no_grad()
@hydra.main(config_path="configs", config_name="predict.yaml", version_base=None)
def predict(cfg: DictConfig):

    head_id_to_entity = ID_MAPPING[cfg.type.head_type]()
    tail_id_to_entity = ID_MAPPING[cfg.type.tail_type]()

    if cfg.verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.ERROR)

    e1_is_all = cfg.e1 == "all"
    e2_is_all = cfg.e2 == "all"

    e1s = get_entity_list(cfg.e1, head_id_to_entity)
    e2s = get_entity_list(cfg.e2, tail_id_to_entity)

    if e1_is_all:
        maybe_mapped_e1s = maybe_mapped_entities(e1s, head_id_to_entity, False)
    else:
        maybe_mapped_e1s = maybe_mapped_entities(e1s, head_id_to_entity, cfg.use_ids)

    if e2_is_all:
        maybe_mapped_e2s = maybe_mapped_entities(e2s, tail_id_to_entity, False)
    else:
        maybe_mapped_e2s = maybe_mapped_entities(e2s, tail_id_to_entity, cfg.use_ids)



    heads = [Entity(cuid, cfg.type.head_type) for cuid in maybe_mapped_e1s]
    tails = [Entity(cuid, cfg.type.tail_type) for cuid in maybe_mapped_e2s]

    if cfg.type.num_workers > 1:
        heads = sorted(heads)
        heads = [head for i, head in enumerate(heads) if i % cfg.num_workers == cfg.type.worker_id]

    processed_pairs = get_processed_pairs(Path(cfg.out))

    if cfg.all_combinations:
        num_queries = len(heads) * len(tails)
    else:
        num_queries = len(heads)
    if num_queries > 100 and not cfg.elastic.server:
        print(f"Using PEDL without a local PubTator copy is only supported for small queries up to 100 protein pairs. Your query contains {len(heads) * len(tails)} pairs. Aborting.")
        sys.exit(1)

    if not cfg.device:
        if torch.cuda.is_available():
            cfg.device = "cuda"
        else:
            cfg.device = "cpu"

    if cfg.elastic.server:
        data_getter = DataGetterPubtator(elasticsearch=cfg.elastic,
                                         entity_marker=cfg.entities.entity_marker,
                                         entity_to_mask=cfg.type.entity_to_mask,
                                         )
    else:
        if cfg.type.head_type == cfg.type.tail_type:
            gene_universe = set(maybe_mapped_e1s + maybe_mapped_e2s)
            chem_universe = None
        else:
            gene_universe = set(maybe_mapped_e2s)
            chem_universe = set(maybe_mapped_e1s)
        data_getter = DataGetterAPI(gene_universe=gene_universe,
                                    chemical_universe=chem_universe,
                                    expand_species=cfg.type.expand_species,
                                    entity_to_mask=cfg.type.entity_to_mask,
                                    entity_marker=cfg.entities.entity_marker
                                    )
    if cfg.skip_processed:
        skip_pairs = processed_pairs
    else:
        skip_pairs = {}

    dataset = PEDLDataset(heads=heads,
                          tails=tails,
                          skip_pairs=skip_pairs,
                          base_model=cfg.type.model_name,
                          data_getter=data_getter,
                          sentence_max_length=500,
                          max_bag_size=cfg.max_bag_size,
                          entity_marker=cfg.entities.entity_marker,
                          max_length=cfg.type.max_sequence_length,
                          label_to_id=cfg.type.label_to_id,
                          entity_to_mask=cfg.type.entity_to_mask,
                          all_combinations=cfg.all_combinations,
                          )
    model = BertForDistantSupervision.from_pretrained(cfg.type.model_name,
                                                      tokenizer=dataset.tokenizer,
                                                      local_model=cfg.type.local_model,
                                                      use_cls=cfg.type.use_cls,
                                                      use_starts=cfg.type.use_starts,
                                                      use_ends=cfg.type.use_ends,
                                                      num_label=cfg.type.num_labels)
    if "cuda" in cfg.device:
        model.bert = nn.DataParallel(model.bert)
    model.eval()
    model.to(cfg.device)
    model.config.e1_id = dataset.tokenizer.convert_tokens_to_ids(cfg.entities.entity_marker["head_start"])
    model.config.e2_id = dataset.tokenizer.convert_tokens_to_ids(cfg.entities.entity_marker["tail_start"])

    os.makedirs(cfg.out, exist_ok=True)

    dataloader = DataLoader(dataset, num_workers=cfg.type.num_workers, batch_size=1,
                            collate_fn=model.collate_fn, prefetch_factor=100)
    with (Path(cfg.out) / f"{PREFIX_PROCESSED_PAIRS}_{uuid.uuid4()}").open("w") as f_pairs_processed:
        for datapoint in tqdm(dataloader, desc="Reading"):
            head, tail = datapoint["pair"]
            if "sentences" not in datapoint:
                f_pairs_processed.write(f"{head}\t{tail}\n")
                continue

            name1 = head.cuid.replace(":", "_")
            name2 = tail.cuid.replace(":", "_")

            file_out = Path(cfg.out) / f"{name1}-_-{name2}.txt"

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
                        label = dataset.get_label(j.item())
                        sentence = datapoint["sentences"][i]
                        f.write(f"{label}\t{max_score.item():.2f}\t{sentence.pmid}\t{sentence.text}\tPEDL\n\n")
            f_pairs_processed.write(f"{head}\t{tail}\n")


def get_entity_list(entity, normalized_entity_ids):
    if isinstance(entity, int):
        entity = str(entity)

    if isinstance(entity, str) and os.path.exists(entity):
        with open(entity) as f:
            p1s = f.read().strip().split("\n")
    else:
        if isinstance(entity, str):
            entity = entity.split()
        if len(entity) == 1 and entity[0] == "all":
            p1s = sorted(normalized_entity_ids.keys())
        else:
            p1s = [str(e) for e in entity]
    return p1s


if __name__ == "__main__":
    predict()