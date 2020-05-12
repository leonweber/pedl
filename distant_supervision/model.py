from torch import nn
import torch
from transformers import BertPreTrainedModel, BertModel
import numpy as np


def aggregate_provenance_predictions(alphas, pmids):
    pmid_predictions = {}
    for alpha, pmid in zip(alphas, pmids):
        pmid = pmid.item()
        if pmid not in pmid_predictions:
            pmid_predictions[pmid] = alpha
        else:
            pmid_predictions[pmid] += alpha

    return pmid_predictions


class BertForDistantSupervision(BertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    def forward(self, token_ids, attention_masks, entity_pos, **kwargs):
        x = self.bert(token_ids, attention_mask=attention_masks)
        pooled_output = x[1]

        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)
        alphas = torch.max(logits, dim=1)[0]
        meta = {
            'alphas': alphas,
            'alphas_by_rel': logits,
            'alphas_hist': np.histogram(alphas.detach().cpu().numpy())
        }

        x = torch.logsumexp(logits, dim=0)

        return x, meta
