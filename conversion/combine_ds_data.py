import argparse
import json
import os
from pathlib import Path



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', nargs='+', type=Path)
    parser.add_argument('output', type=Path)

    args = parser.parse_args()

    combined_data = {}
    for i in args.input:
        with i.open() as f:
            data = json.load(f)
        for k, v in data.items():
            if k in combined_data:
                combined_data[k]['relations'] += v['relations']
                combined_data[k]['mentions'] += v['mentions']
            else:
                combined_data[k] = v

    os.makedirs(args.output.parent, exist_ok=True)
    with args.output.open('w') as f:
        json.dump(combined_data, f)



