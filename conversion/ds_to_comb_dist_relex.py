import argparse
import json
from pathlib import Path

def transform(data, direct_data):
    transformed_data = {}
    for k, v in data.items():
        v['supervision_type'] = 'distant'
        if 'catalysis-precedes' in v['relations']:
            v['relations'].remove('catalysis-precedes')
        transformed_data[k] = v
    for k, v in direct_data.items():
        if k in transformed_data:
            k = k + '_direct'

        v['mentions'] = [m for m in v['mentions'] if m[1] == 'direct']
        if 'catalysis-precedes' in v['relations']:
            v['relations'].remove('catalysis-precedes')
        if v['mentions']:
            v['supervision_type'] = 'direct'
            transformed_data[k] = v

    return transformed_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--direct_data', type=Path)
    parser.add_argument('--pair_blacklist', default=None, type=Path, nargs='*')

    args = parser.parse_args()

    with args.input.open() as f:
        data = json.load(f)

    blacklisted_pairs = set()
    if args.pair_blacklist:
        for path in args.pair_blacklist:
            with path.open() as f:
                blacklisted_pairs.update(json.load(f))

    if args.direct_data:
        with args.direct_data.open() as f:
            direct_data = json.load(f)
            filtered_direct_data = {}
            for k in direct_data:
                if k not in blacklisted_pairs:
                    filtered_direct_data[k] = direct_data[k]
    else:
        direct_data = {}

    with args.output.open('w') as f:
        transformed_data = transform(data, direct_data=direct_data)
        json.dump(transformed_data, f)
