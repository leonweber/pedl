import argparse
import json
import re
import time

import requests
import requests_cache
from pathlib import Path

from tqdm import tqdm


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def get_pmids(data):
    pmids = set()
    for v in data.values():
        pmids.update(m[2].strip() for m in v['mentions'])

    return pmids


def get_year_of_pmids(pmids):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    year_by_pmid = {}
    requests_cache.install_cache(__file__)
    pmids = list(pmids)
    last_time = 0

    for pmid_chunk in tqdm(list(chunks(pmids, 100))):
        raise ValueError("Put your api key here and delete this line")
        data = {'db': 'pubmed', 'id': ','.join(pmid_chunk), 'retmode': 'json',  'api_key': ''}
        elapsed_time = time.time() - last_time
        sleepy_time =  1/8 - elapsed_time
        if sleepy_time > 0:
            time.sleep(sleepy_time)
        last_time = time.time()
        result = requests.get(base_url, params=data)
        result = result.json()['result']
        for pmid in pmid_chunk:
            try:
                year = re.findall(r'\d\d\d\d', result[pmid]['pubdate'])[0]
                year_by_pmid[pmid] = int(year)
            except (KeyError, IndexError):
                continue

    return year_by_pmid



def filter_by_year(data, min_year, max_year):
    pmids = get_pmids(data)
    year_by_pmid = get_year_of_pmids(pmids)
    for k, v in data.items():
        filtered_mentions = []
        for mention in v['mentions']:
            pmid = mention[2].strip()
            if pmid in year_by_pmid and max_year >= year_by_pmid[pmid] >= min_year:
                filtered_mentions.append(mention)
        v['mentions'] = filtered_mentions



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--min_year', type=int, default=0)
    parser.add_argument('--max_year', type=int, default=2100)

    args = parser.parse_args()

    with args.input.open() as f_in, args.output.open('w') as f_out:
        data = json.load(f_in)
        filter_by_year(data, args.min_year, args.max_year)
        json.dump(data, f_out)
