import sys

sys.path.append("/glusterfs/dfs-gfs-dist/barthfab/pedl")

#!/usr/bin/env python

import hydra
from omegaconf import DictConfig
from collections import defaultdict


from tqdm import tqdm

from pedl.utils import Entity
from pedl.data_getter import DataGetterPubtator, DataGetterAPI


@hydra.main(config_path="./configs", config_name="build_training_set.yaml")
def build_training_set(cfg: DictConfig):
    pair_to_relations = defaultdict(set)
    pmid_to_pairs = defaultdict(set)

    gene_universe = set()
    chemical_universe = set()

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
                    gene_universe.add(head.cuid)
                else:
                    raise ValueError(head.type)
                if tail.type == "Chemical":
                    chemical_universe.add(tail.cuid)
                elif tail.type == "Gene":
                    gene_universe.add(tail.cuid)
                else:
                    raise ValueError(tail.type)

    if cfg.chemprot:
        data_getter = DataGetterPubtator(address=cfg.pubtator,
                                         entity_marker=cfg.entities.entity_marker)
        for pair, relations in tqdm(list(pair_to_relations.items()),
                                    desc="Preparing Crawl"):
            head, tail = pair[:2]
            sentences = data_getter.get_sentences(head, tail)
            with open(str(cfg.out) + ".tsv", "w") as f, open(str(cfg.out_blinded) + ".tsv", "w") as f_blinded:
                for sentence in sentences:
                    f.write("\t".join([head.type, head.cuid, tail.type, tail.cuid, ",".join(relations), sentence.text,
                                       sentence.pmid]) + "\n")
                    f_blinded.write("\t".join(
                        [head.type, head.cuid, tail.type, tail.cuid, ",".join(relations), sentence.text_blinded,
                         sentence.pmid]) + "\n")
    else:
        data_getter = DataGetterAPI(chemical_universe=chemical_universe,
                                    gene_universe=gene_universe,
                                    expand_species=cfg.expand_species,
                                    entity_marker=cfg.entities.entity_marker)

        all_pmids = set()

        for pair, relations in tqdm(list(pair_to_relations.items()),
                                    desc="Preparing Crawl"):
            head, tail = pair[:2]
            shared_pmids = data_getter.get_pmids(head) & data_getter.get_pmids(tail)

            all_pmids.update(shared_pmids)
            for pmid in shared_pmids:
                pmid_to_pairs[pmid].add((head, tail))

        with open(str(cfg.out) + "." + str(cfg.worker_id), "w") as f, open(str(cfg.out_blinded) + "." + str(cfg.worker_id), "w") as f_blinded:
            for i, pmid in enumerate(tqdm(sorted(pmid_to_pairs), desc="Crawling")):
                if not (i % cfg.n_worker) == cfg.worker_id:
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


if __name__ == '__main__':
    build_training_set()
