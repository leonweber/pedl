#!/usr/bin/env python

import sys
import hydra
from pathlib import Path
from omegaconf import DictConfig
from pedl import pubtator_elasticsearch


@hydra.main(config_path="configs", config_name="rebuild_pubtator_index.yaml", version_base=None)
def rebuild_pubtator_index(cfg: DictConfig):
    really_continue = input("This will delete the pubtator index and rebuild it. Do you want to continue? Type 'yes':\n")
    if really_continue != "yes":
        sys.exit(0)
    pubtator_elasticsearch.build_index(pubtator_path=Path(cfg.pubtator),
                                       n_processes=cfg.n_processes,
                                       masked_types=cfg.masking.mask_types,
                                       entity_marker=cfg.entities.entity_marker)


if __name__ == '__main__':
    rebuild_pubtator_index()

