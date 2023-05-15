from collections import defaultdict

from torch import nn
import torch
from transformers import BertPreTrainedModel, AutoModel, BertModel, \
    DataCollatorWithPadding
from tqdm import tqdm

from pedl.utils import chunks


class BertForDistantSupervision(BertPreTrainedModel):
    def __init__(self,
                 config,
                 tokenizer,
                 *inputs,
                 use_cls: bool = False,
                 use_starts: bool = False,
                 use_ends: bool = False,
                 entity_marker: dict = None,
                 num_label: int = 7,
                 **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = num_label
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.tokenizer = tokenizer
        self.use_cls = use_cls
        self.use_starts = use_starts
        self.use_ends = use_ends
        self.init_weights()
        if entity_marker:
            self.entity_marker = entity_marker
        else:
            self.entity_marker = {"head_start": '<e1>',
                                  "head_end": '</e1>',
                                  "tail_start": '<e2>',
                                  "tail_end": '</e2>'}
        seq_rep_size = 0
        if use_cls:
            seq_rep_size += self.bert.config.hidden_size
        if use_starts:
            seq_rep_size += 2 * self.bert.config.hidden_size
        if use_ends:
            seq_rep_size += 2 * self.bert.config.hidden_size
        self.classifier = nn.Linear(seq_rep_size, self.num_labels)

    def forward(self, input_ids, attention_mask, use_max=False, **kwargs):
        bert_out = self.bert(input_ids, attention_mask=attention_mask)
        seq_emb = bert_out.last_hidden_state
        seq_reps = []
        if self.use_cls:
            seq_reps.append(seq_emb[:, 0])
        if self.use_starts:
            head_start_idx = torch.where(
                input_ids
                == self.tokenizer.convert_tokens_to_ids(self.entity_marker['head_start'])
            )
            tail_start_idx = torch.where(
                input_ids
                == self.tokenizer.convert_tokens_to_ids(self.entity_marker['tail_start'])
            )
            head_start_rep = seq_emb[head_start_idx]
            tail_start_rep = seq_emb[tail_start_idx]
            start_pair_rep = torch.cat([head_start_rep, tail_start_rep], dim=1)
            seq_reps.append(start_pair_rep)

        if self.use_ends:
            head_end_idx = torch.where(
                input_ids
                == self.tokenizer.convert_tokens_to_ids(self.entity_marker['head_end'])
            )
            tail_end_idx = torch.where(
                input_ids
                == self.tokenizer.convert_tokens_to_ids(self.entity_marker['tail_end'])
            )
            head_end_rep = seq_emb[head_end_idx]
            tail_end_rep = seq_emb[tail_end_idx]
            end_pair_rep = torch.cat([head_end_rep, tail_end_rep], dim=1)
            seq_reps.append(end_pair_rep)

        seq_reps = torch.cat(seq_reps, dim=1)
        seq_emb = self.dropout(seq_reps)

        logits = self.classifier(seq_emb)
        if use_max:
            alphas = torch.max(logits, dim=1)[0]
        else:
            alphas = torch.logsumexp(logits, dim=1)
        meta = {
            'alphas': alphas,
            'alphas_by_rel': logits,
        }

        if use_max:
            bag_logits = torch.max(logits, dim=0)[0]
        else:
            bag_logits = torch.logsumexp(logits, dim=0)

        return bag_logits, meta

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
        for indices_batch in tqdm(list(chunks(indices, batch_size))):
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

