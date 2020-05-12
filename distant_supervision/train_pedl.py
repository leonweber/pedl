import argparse
import json
import logging
import os
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
from transformers import BertConfig
from sklearn.metrics import average_precision_score
from torch import nn
import wandb

from torch.utils.data import DataLoader, RandomSampler, ConcatDataset
from tqdm import trange, tqdm
from transformers import AdamW, WarmupLinearSchedule

from .predict_pedl import predict
from .dataset import DistantBertDataset
from .model import BertForDistantSupervision

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, direct_datasets=None):
    model.train()
    if args.n_gpu > 1 and not hasattr(model.bert, 'module'):
        model.bert = nn.DataParallel(model.bert)
    model.to(args.device)

    if direct_datasets:
        direct_data = ConcatDataset(direct_datasets)
        direct_dataloader = DataLoader(direct_data, batch_size=1 ,shuffle=True)
        direct_iterator = iter(direct_dataloader)
    else:
        direct_iterator = None
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    global_step = 0
    best_val_ap = 0
    loss_fun = nn.BCEWithLogitsLoss()
    direct_loss_fun = nn.BCEWithLogitsLoss()
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    for _ in train_iterator:
        logging_losses = []
        logging_direct_losses = []
        logging_distant_losses = []
        y_pred, y_true = deque(maxlen=100), deque(maxlen=100)
        direct_aps = []
        epoch_iterator = enumerate(train_dataloader)
        pbar = tqdm(total=len(train_dataloader) // args.gradient_accumulation_steps, desc="Batches")
        model.train()
        if args.n_gpu > 1 and not hasattr(model.bert, 'module'):
            model.bert = nn.DataParallel(model.bert)
        model.to(args.device)

        for step, batch in epoch_iterator:
            batch = {k: v.squeeze(0).to(args.device) for k, v in batch.items()}
            logits, meta = model(**batch)

            y_pred.append(logits.cpu().detach().numpy())
            y_true.append(batch['labels'].cpu().numpy())

            distant_loss = loss_fun(logits, batch['labels'].float())
            if direct_iterator:
                distant_loss = (1 - args.direct_weight) * distant_loss

            if args.fp16:
                with amp.scale_loss(distant_loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                distant_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            logging_distant_losses.append(distant_loss.item())

            if direct_iterator:
                try:
                    direct_batch = next(direct_iterator)
                except StopIteration:
                    direct_iterator = iter(direct_dataloader)
                    direct_batch = next(direct_iterator)
                direct_batch = {k: v.squeeze(0).to(args.device) for k, v in direct_batch.items()}
                direct_logits, direct_meta = model(**direct_batch)
                direct_loss = direct_loss_fun(direct_meta['alphas'], direct_batch['is_direct'].float())
                direct_loss = direct_loss + loss_fun(direct_logits, direct_batch['labels'].float())
                direct_loss = args.direct_weight * direct_loss
                logging_direct_losses.append(direct_loss.item())
                direct_ap = average_precision_score(direct_batch['is_direct'].cpu().numpy(), direct_meta['alphas'].cpu().detach().numpy().ravel(),
                                                    average='micro')
                if not np.isnan(direct_ap):
                    direct_aps.append(direct_ap)

                if args.fp16:
                    with amp.scale_loss(direct_loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    direct_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            else:
                direct_loss = 0


            loss = (distant_loss + direct_loss).item()
            logging_losses.append(loss)
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                pbar.update(1)

                ap = average_precision_score(np.vstack(y_true), np.vstack(y_pred), average='micro')
                log_dict = {
                    'loss': np.mean(logging_losses),
                    'direct_loss': np.mean(logging_direct_losses) if logging_direct_losses else None,
                    'distant_loss': np.mean(logging_distant_losses),
                    'distant_ap': ap,
                    'direct_map': np.mean(direct_aps) if direct_aps else None,
                }
                for k, v in meta.items():
                    if hasattr(v, 'detach'):
                        v = v.detach()
                    if hasattr(v, 'cpu'):
                        v = v.cpu().numpy()
                    if not args.disable_wandb:
                        if '_hist' in k:
                            v = wandb.Histogram(np_histogram=v)

                    log_dict[k] = v

                if not args.disable_wandb:
                    wandb.log(log_dict, step=global_step)
                pbar.set_postfix_str(f"loss: {log_dict['loss']}, ap: {ap}, dmAP: {log_dict['direct_map']}")
                logging_losses = []
                logging_direct_losses = []
                logging_distant_losses = []
                direct_aps = []

        # Evaluation
        val_ap = None
        for _, val_ap in predict(dev_dataset, model): # predict yields prediction and current ap => exhaust iterator
            pass
        print()
        print("Validation AP: " + str(val_ap))
        print()
        if not args.disable_wandb:
            wandb.log({'val_distant_ap': val_ap}, step=global_step)


        # Saving
        output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_dir = args.output_dir / f'checkpoint-{global_step}'
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        model.to('cpu')
        if hasattr(model.bert, 'module'):
            model.bert = model.bert.module
        model.save_pretrained(output_dir)

        if val_ap > best_val_ap:
            model.save_pretrained(args.output_dir)
            torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))
            best_val_ap = val_ap



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bert', required=True)
    parser.add_argument('--train', required=True)
    parser.add_argument('--direct_data', default=None, type=Path, nargs='*')
    parser.add_argument('--pair_blacklist', default=None, type=Path, nargs='*')
    parser.add_argument('--dev', required=True)
    parser.add_argument('--seed', default=5005, type=int)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--output_dir', type=Path, default=Path('runs/test'))
    parser.add_argument('--num_train_epochs', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--direct_weight", default=0.0, type=float)
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--max_bag_size", default=None, type=int)
    parser.add_argument("--max_length", default=None, type=int)
    parser.add_argument("--tensor_emb_size", default=200, type=int)
    parser.add_argument("--subsample_negative", default=1.0, type=float)
    parser.add_argument('--ignore_no_mentions', action='store_true')
    parser.add_argument('--init_from', type=Path)
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--disable_wandb', action='store_true')
    parser.add_argument('--test', action='store_true')

    args = parser.parse_args()
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    if not args.disable_wandb:
        wandb.init(project="distant_paths")

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger.warning(f"n_gpu: {args.n_gpu}, 16-bits training: {args.fp16}")

    blacklisted_pairs = set()
    if args.pair_blacklist:
        for path in args.pair_blacklist:
            with path.open() as f:
                blacklisted_pairs.update(json.load(f))

    train_dataset = DistantBertDataset(
        args.train,
        max_bag_size=args.max_bag_size,
        max_length=args.max_length,
        ignore_no_mentions=args.ignore_no_mentions,
        subsample_negative=args.subsample_negative,
        has_direct=False,
        test=args.test
    )
    dev_dataset = DistantBertDataset(
        args.dev,
        max_bag_size=args.max_bag_size,
        max_length=args.max_length,
        ignore_no_mentions=args.ignore_no_mentions,
        has_direct=False,
        test=args.test
    )
    if args.direct_data:
        direct_datasets = []
        for direct_data in args.direct_data:
            direct_datasets.append(DistantBertDataset(
                direct_data,
                max_bag_size=args.max_bag_size,
                max_length=args.max_length,
                ignore_no_mentions=args.ignore_no_mentions,
                has_direct=True,
                pair_blacklist = blacklisted_pairs,
                subsample_negative=0.0,
                test=args.test
            ))
    else:
        direct_datasets = []

    config = BertConfig.from_pretrained(args.bert, num_labels=train_dataset.n_classes )

    model = BertForDistantSupervision.from_pretrained(args.bert,
                                                      config=config
                                                      )
    if not args.disable_wandb:
        wandb.watch(model)
        wandb.config.update(args)
    train(args, train_dataset=train_dataset, model=model, direct_datasets=direct_datasets)
