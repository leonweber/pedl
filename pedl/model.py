from torch import nn
import torch
from transformers import BertPreTrainedModel, BertModel, BertTokenizerFast
import numpy as np


class BertForDistantSupervision(BertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = 7

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size*2, self.num_labels)

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
            'alphas_hist': np.histogram(alphas.detach().cpu().numpy())
        }

        if use_max:
            x = torch.max(logits, dim=0)[0]
        else:
            x = torch.logsumexp(logits, dim=0)

        return x, meta
