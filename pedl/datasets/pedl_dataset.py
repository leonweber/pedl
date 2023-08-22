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
        mask_entities: bool = True,
        sentence_max_length: Optional[int] = None,
        entity_marker: dict = None,
        entity_to_mask: dict = None,
        label_to_id: dict = None,
        all_combinations: bool = True
    ):
        self.label_to_id = label_to_id
        self.id_to_label = {v: k for k, v in label_to_id.items()}

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

        self.tokenizer.add_special_tokens(
            {
                "additional_special_tokens": list(entity_marker.values())
            })
        # if entity_to_mask:
        #     self.tokenizer.add_special_tokens(
        #         {
        #             "additional_special_tokens": [f"<{entity_to_mask['Gene']}{i}/>" for i in range(1, 47)]
        #         }
        #     )

        self.n_classes = len(self.label_to_id)
        self.data_getter = data_getter
        self.relations = relations
        self.mask_entities = mask_entities
        self.sentence_max_length = sentence_max_length
        self.skip_pairs = skip_pairs
        self.max_length = max_length
        self.pair_to_side_information = {}
        if pair_side_information:
            self.pair_to_side_information = self.get_side_information(pair_side_information)
        self.entity_to_side_information = {}
        if entity_side_information:
            self.entity_to_side_information = self.get_side_information(entity_side_information)
        self.all_combinations = all_combinations

        if not self.all_combinations:
            assert len(self.heads) == len(self.tails)


    def __len__(self):
        if self.all_combinations:
            return len(self.heads) * len(self.tails)
        else:
            return len(self.heads)

    def __getitem__(self, idx) -> Optional[Dict]:
        if torch.is_tensor(idx):
            idx = idx.item()

        if self.all_combinations:
            head = self.heads[idx // len(self.tails)]
            tail = self.tails[idx % len(self.tails)]
        else:
            head = self.heads[idx]
            tail = self.tails[idx]

        if (str(head), str(tail)) in self.skip_pairs:
            return {"pair": (head, tail)}

        sentences = self.data_getter.get_sentences(head, tail)

        if self.sentence_max_length:
            sentences = [s for s in sentences if len(s.text) < self.sentence_max_length]

        # deduplicate sentences
        sentences = list(set(sentences))

        if self.relations:
            labels = torch.zeros(len(self.label_to_id))
            for relation in self.relations[idx]:
                labels[self.label_to_id[relation]] = 1
        else:
            labels = None

        if self.max_bag_size and len(sentences) > self.max_bag_size:
            sentences = random.sample(sentences, self.max_bag_size)

        if self.mask_entities:
            texts = [s.text_blinded for s in sentences]
        else:
            texts = [s.text for s in sentences]

        if not texts:
            return {"pair": (head, tail)}

        encoding = self.tokenizer.batch_encode_plus(texts, max_length=self.max_length,
                                                    truncation=True)
        filtered_encoding = {k: [] for k in encoding}
        filtered_sentences = []
        # remove texts with missing entity markers
        for i, text in enumerate(texts):
            if (self.entity_marker["head_start"] in text and self.entity_marker["head_end"] in text and
                    self.entity_marker["tail_start"] in text and self.entity_marker["tail_end"] in text):
                for k in encoding:
                    filtered_encoding[k].append(encoding[k][i])
                filtered_sentences.append(sentences[i])
            else:
                logger.warning(f"Missing entity markers in sentence: {sentences[i]}")

        if len(filtered_sentences) == 0:
            return {"pair": (head, tail)}

        sample = {
            "encoding": filtered_encoding,
            "labels": labels,
            "is_direct": False,
            "sentences": filtered_sentences,
            "pair": (head, tail)
        }

        return sample

    def get_id(self, label):
        return self.label_to_id[label]

    def get_label(self, label_id):
        return self.id_to_label[label_id]

    @staticmethod
    def get_side_information(file_name):
        side_information = {}
        with open(hydra.utils.to_absolute_path(Path("data") / "side_information" / file_name)) as f:
            for line in f:
                cuid, side_info = line.strip("\n").split("\t")
                side_information[cuid] = side_info
        return side_information
