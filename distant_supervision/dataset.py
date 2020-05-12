import logging

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class DistantBertDataset(Dataset):

    def __init__(self, path, max_bag_size=None, max_length=512, ignore_no_mentions=False, subsample_negative=1.0,
                 has_direct=False, pair_blacklist=None, test=False):
        self.file = h5py.File(path, 'r', driver='core')
        self.max_bag_size = max_bag_size
        self.max_length = max_length
        self.pairs = []
        self.entity_ids = self.file['entity_ids'][:]
        self.id2entity = self.file['id2entity'][:]
        for e1_id, e2_id in self.entity_ids:
            e1 = self.id2entity[e1_id].decode()
            e2 = self.id2entity[e2_id].decode()
            self.pairs.append(f"{e1},{e2}")
        self.pairs = np.array(self.pairs)
        self.labels = self.file['labels'][:]
        self.has_direct = has_direct

        if pair_blacklist:
            filtered_pairs = []
            filtered_labels = []
            filtered_entity_ids = []
            for pair, label, entity_id in zip(self.pairs, self.labels, self.entity_ids):
                if pair not in pair_blacklist:
                    filtered_pairs.append(pair)
                    filtered_labels.append(label)
                    filtered_entity_ids.append(entity_id)
            n_removed = len(self.pairs) - len(filtered_pairs)
            self.pairs = filtered_pairs
            self.labels = np.vstack(filtered_labels)
            self.entity_ids = np.vstack(filtered_entity_ids)
            logger.info(f"Removed {n_removed} of {n_removed + len(self.pairs)} pairs due to blacklisting.")


        if ignore_no_mentions:
            filtered_pairs = []
            filtered_labels = []
            filtered_entity_ids = []
            for pair, label, entity_id in zip(self.pairs, self.labels, self.entity_ids):
                if pair in self.file['token_ids']:
                    filtered_pairs.append(pair)
                    filtered_labels.append(label)
                    filtered_entity_ids.append(entity_id)
            self.pairs = filtered_pairs
            self.labels = np.vstack(filtered_labels)
            self.entity_ids = np.vstack(filtered_entity_ids)

        if subsample_negative < 1.0:
            filtered_pairs = []
            filtered_labels = []
            filtered_entity_ids = []
            for pair, label, entity_id in zip(self.pairs, self.labels, self.entity_ids):
                if label.sum() > 0 or np.random.uniform(0, 1) <= subsample_negative:
                    filtered_pairs.append(pair)
                    filtered_labels.append(label)
                    filtered_entity_ids.append(entity_id)
            self.pairs = filtered_pairs
            self.labels = np.vstack(filtered_labels)
            self.entity_ids = np.vstack(filtered_entity_ids)

        self.n_classes = len(self.file['id2label'])
        self.n_entities = len(self.file['id2entity'])

        if test:
            filtered_pairs = []
            filtered_labels = []
            filtered_entity_ids = []
            for pair, label, entity_id in zip(self.pairs, self.labels, self.entity_ids):
                if np.random.uniform(0, 1) <= 0.1:
                    filtered_pairs.append(pair)
                    filtered_labels.append(label)
                    filtered_entity_ids.append(entity_id)
            self.pairs = filtered_pairs
            self.labels = np.vstack(filtered_labels)
            self.entity_ids = np.vstack(filtered_entity_ids)


    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        pair = self.pairs[idx]
        token_ids = self.file.get(f"token_ids/{pair}", np.array([[-1]]))[:]
        attention_masks = self.file.get(f"attention_masks/{pair}", np.array([[-1]]))[:]
        entity_pos = self.file.get(f"entity_positions/{pair}", np.array([[-1]]))[:] # bag_size x e1/e2 x start/end
        is_direct = self.file.get(f"is_direct/{pair}", np.array([[-1]]))[:] # bag_size x e1/e2 x start/end
        pmids = self.file.get(f"pmids/{pair}", np.array([[-1]]))[:] # bag_size x e1/e2 x start/end
        labels = self.labels[idx]
        entity_ids = self.entity_ids[idx]

        token_ids = token_ids[:self.max_bag_size, :self.max_length]
        attention_masks = attention_masks[:self.max_bag_size, :self.max_length]
        entity_pos = entity_pos[:self.max_bag_size]
        is_direct = is_direct[:self.max_bag_size]




        sample = {
            "token_ids": torch.from_numpy(token_ids).long(),
            "attention_masks": torch.from_numpy(attention_masks).long(),
            "entity_pos": torch.from_numpy(entity_pos).long(),
            "entity_ids": torch.from_numpy(entity_ids).long(),
            "labels": torch.from_numpy(labels).long(),
            "is_direct": torch.from_numpy(is_direct).long(),
            "has_mentions": torch.tensor([token_ids[0][0] >= 0]).bool(),
            "pmids": torch.tensor(pmids).long(),
            "has_direct": torch.tensor(self.has_direct)
        }

        return sample