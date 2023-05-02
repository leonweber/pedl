#!/usr/bin/env python

import hydra
from omegaconf import DictConfig
from collections import defaultdict

from tqdm import tqdm

from pedl.utils import Entity, get_hgnc_symbol_to_gene_id
from pedl.data_getter import DataGetterPubtator, DataGetterAPI
from pedl.predict import maybe_mapped_entities


@hydra.main(config_path="configs", config_name="build_training_set.yaml", version_base=None)
def build_training_set(cfg: DictConfig):
    """
`build_training_set` is a function that generates a training set from a dataset of triples containing entity pairs and their relationships. The generated training set consists of sentences from scientific articles mentioning these entity pairs.

**Arguments:**

- `cfg` (`DictConfig`): A dictionary containing the configuration settings for this function. The settings include:
  - `triples`: The file path of the dataset containing triples (entity pairs and their relationships).
  - `elastic`: The ElasticSearch settings.
  - `entities.entity_marker`: The entity marker settings.
  - `out`: The output file path.
  - `out_blinded`: The output file path for blinded sentences (entities replaced by placeholders).

**Outputs:**

This function generates two TSV files:

1. `<cfg.out>.tsv` or `<cfg.out>.<cfg.worker_id>`: A TSV file containing sentences with the following columns:
   - Head entity type
   - Head entity CUID
   - Tail entity type
   - Tail entity CUID
   - Relationships
   - Sentence text
   - PubMed ID

2. `<cfg.out_blinded>.tsv` or `<cfg.out_blinded>.<cfg.worker_id>`: A TSV file containing blinded sentences (entities replaced by placeholders) with the same columns as the first output file.
    """
    pair_to_relations = defaultdict(set)

    gene_universe = set()
    chemical_universe = set()
    gene_mapper = get_hgnc_symbol_to_gene_id()

    with open(cfg.triples) as f:
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
                    head.cuid = gene_mapper.get(head.cuid, head.cuid)
                    gene_universe.add(head.cuid)
                elif head.type == "Protein":
                    head.cuid = gene_mapper.get(head.cuid, head.cuid)
                    gene_universe.add(head.cuid)
                    head.type = "Gene"
                else:
                    raise ValueError(head.type)
                if tail.type == "Chemical":
                    chemical_universe.add(tail.cuid)
                elif tail.type == "Protein":
                    tail.cuid = gene_mapper.get(tail.cuid, tail.cuid)
                    gene_universe.add(tail.cuid)
                    tail.type = "Gene"
                elif tail.type == "Gene":
                    tail.cuid = gene_mapper.get(tail.cuid, tail.cuid)
                    gene_universe.add(tail.cuid)
                else:
                    raise ValueError(tail.type)
    data_getter = DataGetterPubtator(elasticsearch=cfg.elastic,
                                     entity_marker=cfg.entities.entity_marker,
                                     max_size=cfg.max_size,)
    with open(str(cfg.out) + ".tsv", "w") as f, open(str(cfg.out_blinded) + ".tsv", "w") as f_blinded:
        for pair, relations in tqdm(list(pair_to_relations.items()),
                                    desc="Crawling"):
            head, tail = pair[:2]
            sentences = data_getter.get_sentences(head, tail)
            for sentence in sentences:
                f.write("\t".join([head.type, head.cuid, tail.type, tail.cuid, ",".join(relations), sentence.text,
                                   sentence.pmid]) + "\n")
                f_blinded.write("\t".join(
                    [head.type, head.cuid, tail.type, tail.cuid, ",".join(relations), sentence.text_blinded,
                     sentence.pmid]) + "\n")



if __name__ == '__main__':
    build_training_set()
