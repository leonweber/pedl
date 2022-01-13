import pickle
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import hydra.utils
import numpy as np
import torch
import pytorch_lightning as pl
import bioc
import torchmetrics
import transformers
from torch import nn
from tqdm import tqdm
from transformers.file_utils import ModelOutput
import torch.nn.functional as F
from segtok.segmenter import split_single
from transformers.modeling_outputs import SequenceClassifierOutput


class TSVDataset:
    def __init__(self, path, tokenizer,
                 limit_examples, max_length,
                 entity_to_side_information=None,
                 use_none_class=False):
        self.examples = []
        self.meta = utils.get_dataset_metadata(path)
        with open(hydra.utils.to_absolute_path(path)) as f:
            lines = f.readlines()
        if limit_examples:
            lines = lines[:limit_examples]
        for line in tqdm(lines):
            fields = line.strip().split("\t")
            if len(fields) <= 1:
                continue

            type_head, cuid_head, type_tail, cuid_tail, label, text, pmid = fields
            # pair_side_info = pair_to_side_information.get((head.infons["identifier"], tail.infons["identifier"]), "")
            pair_side_info = ""
            head_side_info = entity_to_side_information.get(cuid_head, "")
            tail_side_info = entity_to_side_information.get(cuid_tail, "")
            #
            side_info = f"{pair_side_info} | {head_side_info} | {tail_side_info} [SEP]"

            features_text = tokenizer.encode_plus(
                text=text,  max_length=max_length, truncation="longest_first"
            )
            len_remaining = max_length - len(features_text.input_ids)

            features_side = tokenizer.encode_plus(
                side_info, max_length=len_remaining, truncation="longest_first", add_special_tokens=False
            )

            features = {
                "input_ids": features_text.input_ids + features_side.input_ids,
                "attention_mask": features_text.attention_mask + features_side.attention_mask
            }

            if "token_type_ids" in features_text:
                features["token_type_ids"] = [0] * len(features_text.input_ids) + [1] * len(features_side.input_ids)

            try:
                assert "<e1>" in tokenizer.decode(features["input_ids"])
                assert "</e1>" in tokenizer.decode(features["input_ids"])
                assert "<e2>" in tokenizer.decode(features["input_ids"])
                assert "</e2>" in tokenizer.decode(features["input_ids"])
            except AssertionError:
                log.warning("Truncated entity")
                continue

            features["labels"] = np.zeros(len(self.meta.label_to_id))
            for l in label.split(","):
                features["labels"][self.meta.label_to_id[l]] = 1

            if use_none_class and features["labels"].sum() == 0:
                features["labels"][0] = 1

            self.examples.append({"head": "TODO", "tail": "TODO", "features": features})

    def __getitem__(self, item):
        example = self.examples[item]["features"].copy()
        example["meta"] = self.meta

        return example

    def __len__(self):
        return len(self.examples)
