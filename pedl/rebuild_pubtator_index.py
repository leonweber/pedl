#!/usr/bin/env python

import sys
import hydra
from pathlib import Path
from omegaconf import DictConfig
from pedl import pubtator_elasticsearch


@hydra.main(config_path="configs", config_name="rebuild_pubtator_index.yaml", version_base=None)
def rebuild_pubtator_index(cfg: DictConfig):
    really_continue = input("This will delete the pubtator index and rebuild it. Do you want to continue? Type 'yes':\n").strip()
    if really_continue != "yes":
        print(f"Your input was '{really_continue}', not 'yes'. Aborting.")
        sys.exit(0)
    pubtator_elasticsearch.build_index(pubtator_file=Path(cfg.pubtator_file),
                                       n_processes=cfg.n_processes,
                                       elasticsearch=cfg.elastic,
                                       masked_types=cfg.type.entity_to_mask,
                                       entity_marker=cfg.entities.entity_marker,
                                       )
