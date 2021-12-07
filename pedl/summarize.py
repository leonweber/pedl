#!/usr/bin/env python

import hydra
from omegaconf import DictConfig
from pedl.utils import build_summary_table


@hydra.main(config_path="../configs/summarize", config_name="default.yaml")
def summarize(config: DictConfig):
    if not config.out:
        file_out = (config.path_to_files.parent / config.path_to_files.name).with_suffix(".tsv")
    else:
        file_out = config.out
    with open(file_out, "w") as f:
        f.write(f"p1\tassociation type\tp2\tscore (sum)\tscore (max)\n")
        for row in build_summary_table(config.path_to_files, score_cutoff=config.cutoff,
                                       no_association_type=config.no_association_type):
            f.write(f"{row[0]}\t{row[1]}\t{row[2]}\t{row[3]:.2f}\t{row[4]:.2f}\n")


if __name__ == '__main__':
    summarize()
