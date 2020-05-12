import itertools
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm
import argparse
import json

MAX_BAG_SIZE=100

def get_rel_type(line):
    return line.split('\t')[4].strip()

def get_sup_type(line):
    return line.split('\t')[6].strip()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', required=True, type=Path, nargs='+')
    parser.add_argument('--data', required=True)
    parser.add_argument('--subsample', type=float, default="1.0")
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()


    os.makedirs(args.out.parent, exist_ok=True)

    with open(args.data) as f:
        data = json.load(f)
        files = [f.open() for f in args.raw]

    result = {}
    for triple in data:
        e1, r, e2 = triple.split(',')
        pair = f'{e1},{e2}'
        if pair not in result:
            result[pair] = {
                    'relations': set(),
                    'mentions': set()
                    }
        result[pair]['relations'].add(r)

    for lino, line in tqdm(enumerate(itertools.chain(*files))):
        fields = line.split("\t")
        if len(fields) != 8:
            continue
        e1, e2 = fields[:2]
        pair = f'{e1},{e2}'
        result[pair]['mentions'].add(tuple(fields[5:]))


    with open(args.out, 'w') as f:
        if args.subsample < 1.0:
            pairs = np.empty(len(result), dtype='O')
            pairs[:] = list(result.keys())
            pairs = np.random.choice(pairs, int(len(pairs) * args.subsample), replace=False)
        else:
            pairs = result.keys()

        json_compat_result = {}
        for pair in tqdm(pairs, total=len(pairs)):
            mentions = list(result[pair]['mentions'])
            arr = np.empty(len(mentions), dtype='O')
            arr[:] = mentions
            sampled_mentions = np.random.choice(arr, min(MAX_BAG_SIZE, len(mentions)), replace=False)
            json_compat_result[pair] = result[pair]
            json_compat_result[pair]['mentions'] = sampled_mentions.tolist()
            json_compat_result[pair]['relations'] = list(result[pair]['relations'])
        
        json.dump(json_compat_result, f, indent=2)

