import argparse
import json
import os
import re
from collections import deque, defaultdict
from glob import glob
from pathlib import Path
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import numpy as np
from transformers import WEIGHTS_NAME

from .dataset import DistantBertDataset
from .model import BertForDistantSupervision



def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)


def predict(dataset, model, data=None):
    model.eval()
    dataloader = DataLoader(dataset,  batch_size=1)
    data_it = tqdm(dataloader, desc="Predicting", total=len(dataset))
    y_pred, y_true = deque(), deque()

    for batch in data_it:
        model.eval()
        batch = {k: v.squeeze(0).to('cuda') for k, v in batch.items()}
        with torch.no_grad():
            logits, meta = model(**batch)

        e1, e2 = batch['entity_ids']


        e1 = dataset.file['id2entity'][e1].decode()
        e2 = dataset.file['id2entity'][e2].decode()

        assert f"{e1},{e2}" in dataset.pairs

        prediction = {}
        prediction['entities'] = [e1, e2]

        prediction['labels'] = []
        prediction['true_labels'] = []
        prediction['alphas'] = torch.sigmoid(meta['alphas']).tolist()
        alphas_by_rel = torch.sigmoid(meta['alphas_by_rel'])
        if data:
            prediction['mentions'] = data[f"{e1},{e2}"]['mentions']
        ap = None
        prediction['alphas_by_rel'] = {}
        for i, logit in enumerate(logits):
            rel = dataset.file['id2label'][i].decode()
            score = torch.sigmoid(logit).item()
            prediction['labels'].append([rel, score])
            prediction['alphas_by_rel'][rel] = alphas_by_rel[:, i].tolist()

        if 'labels' in batch:
            y_pred.append(logits.cpu().detach().numpy())
            y_true.append(batch['labels'].cpu().numpy())
            ap = average_precision_score(np.vstack(y_true), np.vstack(y_pred), average='micro')
            data_it.set_postfix_str(f"ap: {ap}")

            for i, label in enumerate(batch['labels']):
                if label.item() > 0:
                    rel = dataset.file['id2label'][i].decode()
                    prediction['true_labels'].append(rel)

        yield prediction, ap




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--model_path', required=True, type=Path)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--device', default='cpu')

    args = parser.parse_args()

    dataset = DistantBertDataset(
        args.input,
        # max_bag_size=train_args.max_bag_size,
        # max_length=train_args.max_length,
        # ignore_no_mentions=train_args.ignore_no_mentions
        max_bag_size=100,
        max_length=None,
        ignore_no_mentions=True
    )

    model = BertForDistantSupervision.from_pretrained(args.model_path)
    model.bert = nn.DataParallel(model.bert)
    model.to(args.device)

    with args.data.open() as f:
        data = json.load(f)

    checkpoints = list(os.path.dirname(c) for c in natural_sort(glob(str(args.model_path / '**' / WEIGHTS_NAME), recursive=True))[::-1])

    best_ap = (None, 0)
    for checkpoint in checkpoints:
        model = BertForDistantSupervision.from_pretrained(checkpoint)
        model.parallel_bert = nn.DataParallel(model.bert)
        model.to(args.device)
        with args.output.open('w') as f:

            for prediction, ap in predict(dataset=dataset, model=model, data=data):
                f.write(json.dumps(prediction) + "\n")
            if ap > best_ap[1]:
                best_ap = (checkpoint, ap)
    print(best_ap)



