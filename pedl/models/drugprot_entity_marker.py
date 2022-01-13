from collections import defaultdict
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import hydra.utils
import torch
import pytorch_lightning as pl
import torchmetrics
import transformers
from torch import nn
from transformers.file_utils import ModelOutput
import torch.nn.functional as F
from transformers.modeling_outputs import SequenceClassifierOutput

from pedl.utils import chunks, DatasetMetaInformation, get_logger


log = get_logger(__name__)


@dataclass
class MultitaskClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    dataset_to_logits: Dict[str, torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class ATLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels):
        # TH label
        th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
        th_label[:, 0] = 1.0
        labels[:, 0] = 0.0

        p_mask = labels + th_label
        n_mask = 1 - labels

        # Rank positive classes to TH
        logit1 = logits - (1 - p_mask) * 1e30
        loss1 = -(F.log_softmax(logit1, dim=-1) * labels).sum(1)

        # Rank TH to negative classes
        logit2 = logits - (1 - n_mask) * 1e30
        loss2 = -(F.log_softmax(logit2, dim=-1) * th_label).sum(1)

        # Sum two parts
        loss = loss1 + loss2
        loss = loss.mean()
        return loss

    def get_label(self, logits, num_labels=-1):
        th_logit = logits[:, 0].unsqueeze(1)
        output = torch.zeros_like(logits).to(logits)
        mask = logits > th_logit
        if num_labels > 0:
            top_v, _ = torch.topk(logits, num_labels, dim=1)
            top_v = top_v[:, -1]
            mask = (logits >= top_v.unsqueeze(1)) & mask
        output[mask] = 1.0
        output[:, 0] = (output.sum(1) == 0.0).to(logits)
        return output


class EntityMarkerBaseline(pl.LightningModule):
    def __init__(
        self,
        transformer: str,
        lr: float,
        finetune_lr: float,
        loss: str,
        tune_thresholds: bool,
        use_doc_context: bool,
        dataset_to_meta: Dict[str, DatasetMetaInformation],
        max_length: int,
        optimized_metric: str,
        use_cls: bool = False,
        use_starts: bool = False,
        use_ends: bool = False,
        mark_with_special_tokens: bool = True,
        blind_entities: bool = False,
        entity_side_information = None,
        pair_side_information = None,
        use_none_class = True,
        entity_embeddings = None,
        weight_decay=0.0
    ):
        super().__init__()

        self.weight_decay = weight_decay
        self.use_none_class = use_none_class
        self.entity_side_information = {}
        if entity_side_information is not None:
            with open(hydra.utils.to_absolute_path(Path("data") / "side_information" / entity_side_information)) as f:
                for line in f:
                    cuid, side_info = line.strip("\n").split("\t")
                    self.entity_side_information[cuid] = side_info

        self.pair_side_information = {}
        if pair_side_information is not None:
            with open(hydra.utils.to_absolute_path(Path("data") / "side_information" / pair_side_information)) as f:
                for line in f:
                    cuid_head, cuid_tail, side_info = line.strip("\n").split("\t")
                    self.pair_side_information[(cuid_head, cuid_tail)] = side_info

        self.use_cls = use_cls
        self.max_length = max_length
        self.optimized_metric = optimized_metric
        self.use_starts = use_starts
        self.use_ends = use_ends
        self.mark_with_special_tokens = mark_with_special_tokens
        self.blind_entities = blind_entities

        assert use_cls or use_starts or use_ends

        if not mark_with_special_tokens:
            assert (
                not use_starts and not use_ends
            ), "Starts and ends cannot be uniquely determined without additional special tokens"

        self.dataset_to_meta = dataset_to_meta
        loss = loss.lower().strip()
        assert loss in {"atlop", "bce"}
        if loss == "atlop":
            self.loss = ATLoss()
        else:
            self.loss = nn.BCEWithLogitsLoss()

        self.tune_thresholds = tune_thresholds
        if self.tune_thresholds and isinstance(self.loss, ATLoss):
            warnings.warn(
                "atlop loss has no fixed thresholds. Setting tune_thresholds to False"
            )
            self.tune_thresholds = False
        self.use_doc_context = use_doc_context

        self.transformer = transformers.AutoModel.from_pretrained(transformer)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(transformer)

        if mark_with_special_tokens:
            self.tokenizer.add_tokens(
               ['<e1>', '</e1>', '<e2>', '</e2>'], special_tokens=True
            )
            self.transformer.resize_token_embeddings(len(self.tokenizer))
        self.dropout = nn.Dropout(self.transformer.config.hidden_dropout_prob)

        self.dataset_to_train_f1 = {}
        self.dataset_to_dev_f1 = {}
        seq_rep_size = 0
        if use_cls:
            seq_rep_size += self.transformer.config.hidden_size
        if use_starts:
            seq_rep_size += 2 * self.transformer.config.hidden_size
        if use_ends:
            seq_rep_size += 2 * self.transformer.config.hidden_size
        if entity_embeddings:
            entity_embeddings = Path(entity_embeddings)
            with open(entity_embeddings / "embeddings.pkl", "rb") as f:
                embeddings = pickle.load(f)
                self.entity_embeddings = nn.Embedding(embeddings.shape[0] + 1, embeddings.shape[1])
                with torch.no_grad():
                    self.entity_embeddings.weight[0, :] = 0
                    self.entity_embeddings.weight[1:] = nn.Parameter(embeddings)
                self.entity_embeddings.requires_grad = False
                self.entity_mlp = nn.Sequential(nn.Linear(self.entity_embeddings.embedding_dim*2, 100), nn.ReLU(), nn.Dropout(0.5), nn.Linear(100, 100), nn.Dropout(0.5))
                seq_rep_size += 100
            self.entity_to_embedding_index = {}
            with open(entity_embeddings / "entities.dict") as f:
                for line in f:
                    fields = line.strip().split("\t")
                    index = int(fields[1])
                    self.entity_to_embedding_index[fields[0]] = index
        else:
            self.entity_embeddings = None
            self.entity_to_embedding_index = None


        self.meta = dataset_to_meta["drugprot"]
        self.classifier = nn.Linear(
            seq_rep_size, len(self.meta.label_to_id)
        )
        for dataset, meta in dataset_to_meta.items():
            self.dataset_to_train_f1[dataset] = torchmetrics.F1(
                num_classes=len(self.meta.label_to_id) - 1
            )
            self.dataset_to_dev_f1[dataset] = torchmetrics.F1(
                num_classes=len(self.meta.label_to_id) - 1
            )


        self.lr = lr
        self.finetune_lr = finetune_lr
        self.num_training_steps = None

    def collate_fn(self, data):
        meta = data[0].pop("meta")
        for i in data[1:]:
            m = i.pop("meta")
            assert m.label_to_id == meta.label_to_id
        collator = transformers.DataCollatorWithPadding(self.tokenizer)
        batch = collator(data)
        batch["meta"] = meta

        return batch

    def forward(self, features):
        if "token_type_ids" in features:
            output = self.transformer(
                input_ids=features["input_ids"],
                token_type_ids=features["token_type_ids"],
                attention_mask=features["attention_mask"],
            )
        else:
            output = self.transformer(
                input_ids=features["input_ids"],
                attention_mask=features["attention_mask"],
            )
        seq_emb = self.dropout(output.last_hidden_state)

        seq_reps = []

        if self.use_cls:
            seq_reps.append(seq_emb[:, 0])
        if self.use_starts:
            head_start_idx = torch.where(
                features["input_ids"]
                == self.tokenizer.convert_tokens_to_ids('<e1>')
            )
            tail_start_idx = torch.where(
                features["input_ids"]
                == self.tokenizer.convert_tokens_to_ids('<e2>')
            )
            head_start_rep = seq_emb[head_start_idx]
            tail_start_rep = seq_emb[tail_start_idx]
            start_pair_rep = torch.cat([head_start_rep, tail_start_rep], dim=1)
            seq_reps.append(start_pair_rep)

        if self.use_ends:
            head_end_idx = torch.where(
                features["input_ids"]
                == self.tokenizer.convert_tokens_to_ids('</e1>')
            )
            tail_end_idx = torch.where(
                features["input_ids"]
                == self.tokenizer.convert_tokens_to_ids('</e2>')
            )
            head_end_rep = seq_emb[head_end_idx]
            tail_end_rep = seq_emb[tail_end_idx]
            end_pair_rep = torch.cat([head_end_rep, tail_end_rep], dim=1)
            seq_reps.append(end_pair_rep)

        seq_reps = torch.cat(seq_reps, dim=1)
        if self.entity_embeddings:
            e1_embeddings = self.entity_embeddings(features["e1_embedding_index"])
            e2_embeddings = self.entity_embeddings(features["e2_embedding_index"])
            pair_embeddings = self.entity_mlp(torch.cat([e1_embeddings, e2_embeddings], dim=1))
            seq_reps = torch.cat([seq_reps, pair_embeddings], dim=1)
        datset_type = features["meta"].type

        logits = self.classifier(seq_reps)
        if "labels" in features:
            if datset_type == "distant":
                pooled_logits = torch.logsumexp(logits, dim=1)
                loss = self.loss(pooled_logits, torch.ones_like(pooled_logits))
            else:
                loss = self.loss(logits, features["labels"])
        else:
            loss = None

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
        )

    def forward_batched(self, features, batch_size, **kwargs):
        logits_all = []
        meta_all = defaultdict(list)

        if not batch_size:
            return self.forward(features)

        indices = torch.arange(len(features["input_ids"]))
        for indices_batch in chunks(indices, batch_size):
            if "token_type_ids" in features:
                batch_features = {"input_ids": features["input_ids"][indices_batch],
                                  "attention_mask": features["attention_mask"][indices_batch],
                                  "token_type_ids": features["token_type_ids"][indices_batch]}
            else:
                batch_features = {"input_ids": features["input_ids"][indices_batch],
                                  "attention_mask": features["attention_mask"][indices_batch]}
            logits, meta = self.forward(batch_features)
            logits_all.append(logits)
            for k, v in meta.items():
                meta_all[k].append(v)

        logits_all = torch.cat(logits_all)
        for k, v in meta_all.items():
            meta_all[k] = torch.cat(v)

        return logits_all, meta_all
