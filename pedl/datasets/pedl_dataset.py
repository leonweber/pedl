import logging
import random
from pathlib import Path
from typing import List, Tuple, Set, Optional, Dict

import hydra.utils
import torch
from torch.utils.data import Dataset, Sampler
from transformers import AutoTokenizer
from segtok.segmenter import split_single

from pedl.utils import Entity
from pedl.data_getter import DataGetterAPI

logger = logging.getLogger(__name__)


class PEDLDataset(Dataset):
    label_to_id = {
        "in-complex-with": 0,
        "controls-state-change-of": 1,
        "controls-transport-of": 2,
        "controls-phosphorylation-of": 3,
        "controls-expression-of": 4,
        "catalysis-precedes": 5,
        "interacts-with": 6,
    }
    id_to_label = {v: k for k, v in label_to_id.items()}

    def __init__(
        self,
        heads: List[Entity],
        tails: List[Entity],
        skip_pairs: Set[Tuple[str, str]],
        base_model: str,
        data_getter: DataGetterAPI,
        max_length: int,
        pair_side_information: str = None,
        entity_side_information: str = None,
        relations: Optional[List[Set[str]]] = None,
        max_bag_size: Optional[int] = None,
        blind_entities: bool = True,
        sentence_max_length: Optional[int] = None,
        entity_marker: dict = None,
        masking_types: dict = None,
    ):
        self.heads = heads
        self.tails = tails
        self.max_bag_size = max_bag_size
        self.tokenizer = AutoTokenizer.from_pretrained(str(base_model))
        if entity_marker:
            self.entity_marker = entity_marker
        else:
            self.entity_marker = {"head_start": '<e1>',
                                  "head_end": '</e1>',
                                  "tail_start": '<e2>',
                                  "tail_end": '</e2>'}
        if masking_types:
            assert masking_types["Gene"], "By now only entity masking for proteins is implemented. Please use Gene "
            self.tokenizer.add_special_tokens(
                {
                    "additional_special_tokens": list(entity_marker.values()) + [f"<protein{i}/>" for i in range(1, 47)]
                }
            )
        else:
            self.tokenizer.add_special_tokens(
                {
                    "additional_special_tokens": list(entity_marker.values())
                })
        self.n_classes = len(self.label_to_id)
        self.data_getter = data_getter
        self.relations = relations
        self.blind_entities = blind_entities
        self.sentence_max_length = sentence_max_length
        self.skip_pairs = skip_pairs
        self.max_length = max_length
        self.pair_to_side_information = {}
        if pair_side_information:
            self.pair_to_side_information = self.get_side_information(pair_side_information)
        self.entity_to_side_information = {}
        if entity_side_information:
            self.entity_to_side_information = self.get_side_information(entity_side_information)

    def __len__(self):
        return len(self.heads) * len(self.tails)

    def __getitem__(self, idx) -> Optional[Dict]:
        if torch.is_tensor(idx):
            idx = idx.item()

        head = self.heads[idx // len(self.tails)]
        tail = self.tails[idx % len(self.tails)]

        if (str(head), str(tail)) in self.skip_pairs:
            return {"pair": (head, tail)}

        sentences = self.data_getter.get_sentences(head, tail)

        if self.sentence_max_length:
            sentences = [s for s in sentences if len(s.text) < self.sentence_max_length]

        if not sentences:
            return {"pair": (head, tail)}

        if self.relations:
            labels = torch.zeros(len(self.label_to_id))
            for relation in self.relations[idx]:
                labels[self.label_to_id[relation]] = 1
        else:
            labels = None

        if self.max_bag_size and len(sentences) > self.max_bag_size:
            sentences = random.sample(sentences, self.max_bag_size)

        if self.blind_entities:
            texts = [s.text_blinded for s in sentences]
        else:
            texts = [s.text for s in sentences]

        if self.pair_to_side_information or self.entity_to_side_information:
            encoding = []
            pair_side_info = self.pair_to_side_information.get((head.infons["identifier"], tail.infons["identifier"]), "")

            head_side_info = self.entity_to_side_information.get(head.infons["identifier"], "")
            tail_side_info = self.entity_to_side_information.get(tail.infons["identifier"], "")

            if head_side_info and tail_side_info:
                head_side_info = split_single(head_side_info)[0]
                tail_side_info = split_single(tail_side_info)[0]

            side_info = f"{pair_side_info} | {head_side_info} | {tail_side_info} [SEP]"
            for sentence in texts:
                features_text = self.tokenizer.encode_plus(
                    text=sentence, max_length=self.max_length, truncation="longest_first"
                )
                len_remaining = self.max_length - len(features_text.input_ids)
                features_side = self.tokenizer.encode_plus(
                    side_info, max_length=len_remaining, truncation="longest_first", add_special_tokens=False
                )
                encoding.append(features_text.input_ids + features_side.input_ids)
        else:
            encoding = self.tokenizer.batch_encode_plus(texts, max_length=self.max_length,
                                                        truncation=True)
        sample = {
            "encoding": encoding,
            "labels": labels,
            "is_direct": False,
            "sentences": sentences,
            "pair": (head, tail)
        }

        return sample

    @staticmethod
    def get_side_information(file_name):
        side_information = {}
        with open(hydra.utils.to_absolute_path(Path("data") / "side_information" / file_name)) as f:
            for line in f:
                cuid, side_info = line.strip("\n").split("\t")
                side_information[cuid] = side_info
        return side_information
