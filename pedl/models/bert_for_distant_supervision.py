from collections import defaultdict

from torch import nn
import torch
from transformers import BertPreTrainedModel, BertModel, BertTokenizerFast, \
    DataCollatorWithPadding
import numpy as np

from pedl.utils import chunks


class BertForDistantSupervision(BertPreTrainedModel):
    def __init__(self, config, tokenizer, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = 7

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size*2, self.num_labels)
        self.tokenizer = tokenizer

        self.init_weights()

    def forward(self, input_ids, attention_mask, use_max=False, **kwargs):
        bert_out = self.bert(input_ids, attention_mask=attention_mask)
        x = bert_out.last_hidden_state
        e1_mask = (input_ids == self.config.e1_id).long()
        e2_mask = (input_ids == self.config.e2_id).long()
        e1_idx = e1_mask.argmax(dim=1)
        e2_idx = e2_mask.argmax(dim=1)
        e1_idx[e1_mask.sum(dim=1) == 0] = 0 # default to [CLS] if entity was truncated
        e2_idx[e2_mask.sum(dim=1) == 0] = 0 # default to [CLS] if entity was truncated
        e1_embs = x[torch.arange(len(e1_idx)), e1_idx]
        e2_embs = x[torch.arange(len(e2_idx)), e2_idx]
        x = torch.cat([e1_embs, e2_embs], dim=1)
        x = self.dropout(x)

        logits = self.classifier(x)
        if use_max:
            alphas = torch.max(logits, dim=1)[0]
        else:
            alphas = torch.logsumexp(logits, dim=1)
        meta = {
            'alphas': alphas,
            'alphas_by_rel': logits,
        }

        if use_max:
            x = torch.max(logits, dim=0)[0]
        else:
            x = torch.logsumexp(logits, dim=0)

        return x, meta

    def collate_fn(self, batch):
        if "sentences" not in batch[0]:
            return batch[0]
        collated_batch = {}
        collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        for k, v in batch[0].items():
            if k != "encoding":
                collated_batch[k] = v
        collated_batch["encoding"] = collator(batch[0]["encoding"])

        return collated_batch

    def forward_batched(self, input_ids, attention_mask, batch_size, **kwargs):
        logits_all = []
        meta_all = defaultdict(list)

        if not batch_size:
            return self.forward(input_ids, attention_mask)

        indices = torch.arange(len(input_ids))
        for indices_batch in chunks(indices, batch_size):
            input_ids_batch = input_ids[indices_batch]
            attention_mask_batch = attention_mask[indices_batch]
            logits, meta = self.forward(input_ids=input_ids_batch,
                                        attention_mask=attention_mask_batch)
            logits_all.append(logits)
            for k, v in meta.items():
                meta_all[k].append(v)

        logits_all = torch.cat(logits_all)
        for k, v in meta_all.items():
            meta_all[k] = torch.cat(v)

        return logits_all, meta_all

