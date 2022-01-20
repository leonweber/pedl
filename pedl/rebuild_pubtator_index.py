#!/usr/bin/env python

import sys
import hydra
from omegaconf import DictConfig
from pedl import pubtator_elasticsearch


@hydra.main(config_path="../configs/rebuild_pubtater_index", config_name="default.yaml")
def rebuild_pubtator_index(cfg: DictConfig):
    really_continue = input("This will delete the pubtator index and rebuild it. Do you want to continue? Type 'yes':\n")
    if really_continue != "yes":
        sys.exit(0)
    pubtator_elasticsearch.build_index(pubtator_path=cfg.pubtator,
                                       n_processes=cfg.n_processes)


if __name__ == '__main__':
    rebuild_pubtator_index()
