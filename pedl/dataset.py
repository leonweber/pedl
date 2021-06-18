import json
import logging
import random
from collections import Counter

import torch
from torch.utils.data import Dataset, Sampler
from transformers import BertTokenizerFast

logger = logging.getLogger(__name__)



class PEDLDataset(Dataset):

    label_to_id = {'in-complex-with': 0, 'controls-state-change-of': 1,  'controls-transport-of': 2,
                        'controls-phosphorylation-of': 3, 'controls-expression-of': 4, 'catalysis-precedes': 5, 'interacts-with': 6}
    id_to_label = {v: k for k, v in label_to_id.items()}

    def __init__(self, path, bert, max_bag_size=50, max_length=512, subsample_mentions=False, pair_blacklist=None):
        with open(path) as f:
            self.data = json.load(f)
        self.data = {k: v for k, v in self.data.items() if v["mentions"]}
        self.max_bag_size = max_bag_size
        self.max_length = max_length
        self.pairs = []
        self.tokenizer = BertTokenizerFast.from_pretrained(str(bert))
        self.tokenizer.add_special_tokens({ 'additional_special_tokens': ['<e1>','</e1>', '<e2>', '</e2>'] +
                                                                    [f'<protein{i}/>' for i in range(1, 47)]})
        self.e1_id = self.tokenizer.convert_tokens_to_ids("<e1>")
        self.e2_id = self.tokenizer.convert_tokens_to_ids("<e2>")
        self.subsample_mentions = subsample_mentions

        self.entity_ids = set()
        for pair in self.data:
            self.entity_ids.update(pair.split(","))
        self.entity_ids = sorted(self.entity_ids)

        self.pairs = sorted(self.data)

        if pair_blacklist:
            filtered_pairs = []
            for pair, label, entity_id in zip(self.pairs, self.labels, self.entity_ids):
                if pair not in pair_blacklist:
                    filtered_pairs.append(pair)
            n_removed = len(self.pairs) - len(filtered_pairs)
            self.pairs = filtered_pairs
            logger.info(f"Removed {n_removed} of {n_removed + len(self.pairs)} pairs due to blacklisting.")


        self.n_classes = len(self.label_to_id)
        self.n_entities = len(self.entity_ids)

        self.label_count = Counter()
        for v in self.data.values():
            self.label_count.update([i for i in v["relations"] if i != "NA"])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        pair = self.pairs[idx]
        encoding, is_direct, pmids, mentions = self.get_encoding_plus_meta_info(pair)
        labels = self.get_labels(pair)

        input_ids = torch.tensor(encoding["input_ids"])
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)

        attention_mask = torch.tensor(encoding["attention_mask"])
        if attention_mask.ndim == 1:
            attention_mask = attention_mask.unsqueeze(0)

        entity_ids = pair.split(",")
        entity_ids = [",".join(entity_ids[:-1]),  entity_ids[-1]]
        sample = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "is_direct": torch.tensor(is_direct),
            "pmids": pmids,
            "entity_ids": entity_ids,
            "mentions": mentions
        }

        return sample

    def get_encoding_plus_meta_info(self, pair):
        key = "masked_mention" if "masked_mention" in self.data[pair] else "masked_mentions"
        direct_indices = set()
        distant_indices = set()
        for i, (_, direct_or_distant, _) in enumerate(self.data[pair][key]):
            if direct_or_distant == "direct":
                direct_indices.add(i)
            else:
                distant_indices.add(i)
        sampled_indices = set()
        if self.subsample_mentions and len(self.data[pair][key]) > self.max_bag_size:
            sampled_indices.update(random.sample(direct_indices, k=self.max_bag_size))

        n_free = self.max_bag_size - len(sampled_indices)

        if n_free > 0:
            if self.subsample_mentions and len(distant_indices) > n_free:
                sampled_indices += random.sample(distant_indices, k=n_free)
            else:
                sampled_indices.update(distant_indices)

        masked_mentions = [d for i, d in enumerate(self.data[pair][key]) if i in sampled_indices]
        mentions = [d for i, d in enumerate(self.data[pair]['mentions']) if i in sampled_indices]
        encoding = self.tokenizer.batch_encode_plus([i[0] for i in masked_mentions], pad_to_max_length=True,
                                                    max_length=self.max_length)
        is_direct = [i[1] == "direct" for i in masked_mentions]
        pmids = [i[2] for i in masked_mentions]

        return encoding, is_direct, pmids, mentions

    def get_labels(self, pair):
        labels = torch.zeros(self.n_classes)
        for rel in self.data[pair]["relations"]:
            if rel not in {"NA", "foo"}:
                labels[self.label_to_id[rel]] = 1
        return labels

