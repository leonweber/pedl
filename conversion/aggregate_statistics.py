import argparse
from collections import defaultdict
from pathlib import Path


def aggregate(files):
    min_dist = defaultdict(lambda: float('inf'))
    min_pmid = defaultdict(lambda: None)
    for file in files:
        with file.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                pair, pmid, dist = line.split('\t')
                dist = int(dist)
                if dist < min_dist[pair]:
                    min_dist[pair] = dist
                    min_pmid[pair] = pmid

    for pair in min_dist:
        yield "\t".join([pair, min_pmid[pair], str(min_dist[pair])]) + "\n"






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)

    args = parser.parse_args()

    input_files = args.input.glob('statistics.tsv*')
    with args.output.open('w') as f:
        f.writelines(aggregate(input_files))



