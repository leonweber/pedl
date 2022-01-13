#!/usr/bin/env python

import hydra
from omegaconf import DictConfig
from pedl.utils import build_summary_table


@hydra.main(config_path="../configs/summarize", config_name="default.yaml")
def summarize(cfg: DictConfig):
    if not cfg.out:
        file_out = (cfg.path_to_files.parent / cfg.path_to_files.name).with_suffix(".tsv")
    else:
        file_out = cfg.out
    with open(file_out, "w") as f:
        f.write(f"p1\tassociation type\tp2\tscore (sum)\tscore (max)\n")
        for row in build_summary_table(cfg.path_to_files, score_cutoff=cfg.cutoff,
                                       no_association_type=cfg.no_association_type):
            f.write(f"{row[0]}\t{row[1]}\t{row[2]}\t{row[3]:.2f}\t{row[4]:.2f}\n")


if __name__ == '__main__':
    summarize()
