import json
import logging
import random
from collections import Counter
from pathlib import Path
from typing import List, Tuple, Set, Optional, Dict

import torch
from torch.utils.data import Dataset, Sampler
from transformers import BertTokenizerFast

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
        pairs: List[Tuple[Entity, Entity]],
        base_model: str,
        data_getter: DataGetterAPI,
        relations: Optional[List[Set[str]]] = None,
        max_bag_size: Optional[int] = None,
        blind_entities: bool = True,
        sentence_max_length: Optional[int] = None
    ):
        self.max_bag_size = max_bag_size
        self.pairs = pairs
        self.tokenizer = BertTokenizerFast.from_pretrained(str(base_model))
        self.tokenizer.add_special_tokens(
            {
                "additional_special_tokens": [
                    "<e1>",
                    "</e1>",
                    "<e2>",
                    "</e2>",
                ]
                + [f"<protein{i}/>" for i in range(1, 47)]
            }
        )
        self.n_classes = len(self.label_to_id)
        self.data_getter = data_getter
        self.relations = relations
        self.blind_entities = blind_entities
        self.sentence_max_length = sentence_max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx) -> Optional[Dict]:
        if torch.is_tensor(idx):
            idx = idx.item()

        head, tail = self.pairs[idx]
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

        encoding = self.tokenizer.batch_encode_plus(texts, max_length=312,
                                                    truncation=True)

        sample = {
            "encoding": encoding,
            "labels": labels,
            "is_direct": False,
            "sentences": sentences,
            "pair": (head, tail)
        }

        return sample
