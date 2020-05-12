import os
import json
from pathlib import Path
from collections import defaultdict
from argparse import ArgumentParser


from pairs import PairGetter

from gen_ann_file import get_augmented_offset_lines, load_annotations
from tax_ids import TAX_IDS


TypedEntity = PairGetter.TypedEntity



def write_examples(getter: PairGetter, offset_lines, fname):
    with open(fname,  'w') as f:
        for i, doc in enumerate(getter.get_relevant_docs(offset_lines)):

            if args.test and i % 100 == 0:
                print(f"TEST RUN: {i}/1000 ")
            if args.test and i > 1000:
                break

            for pair in getter._pmid_to_entity_sets[doc.pmid]:
                distance = getter.get_distance(pair, doc)
                f.write(f"{pair[0].id},{pair[1].id}\t{doc.pmid}\t{distance}\n")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--offsets', required=True, type=Path)
    parser.add_argument('--pmc_dir', default='data/pmc_bioc', type=Path)
    parser.add_argument('--worker', default='0', type=int)
    parser.add_argument('--n_workers', default='1', type=int)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--species', default="")
    parser.add_argument('--mapping', default=None)


    args = parser.parse_args()


    pairs = set()
    with open(args.data) as f:
        triples = json.load(f)
        train_triples = []
        items = list(triples.items())
        items = [(k,v) for k,v in items if k.split(',')[1] != 'NA']
        for triple, pmids in items:
            triple = tuple(triple.split(",") + [pmids])
            train_triples.append(triple)
            pairs.add((TypedEntity(triple[0], 'Gene'), TypedEntity(triple[2], 'Gene')))


    species = [TAX_IDS[s] for s in args.species.split(',')]

    mapping = {}
    if args.mapping:
        with open(args.mapping) as f:
            mapping['Gene'] = json.load(f)

    with open(args.offsets) as f:
        offset_lines = get_augmented_offset_lines(f, pmc_dir=args.pmc_dir, types={'Gene'}, mapping=mapping,
                                                  homologue_species=species,
                                                  test=args.test, worker=args.worker, n_workers=args.n_workers)
        anns = load_annotations(offset_lines)
        getter = PairGetter(entity_sets=pairs, anns=anns)



    os.makedirs(args.out.parent, exist_ok=True)

    with open(args.offsets) as f:
        offset_lines = get_augmented_offset_lines(f, pmc_dir=args.pmc_dir, types={'Gene'}, mapping=mapping,
                                                  homologue_species=species,
                                                  test=args.test, worker=args.worker, n_workers=args.n_workers)
        write_examples(getter=getter, offset_lines=offset_lines, fname=str(args.out) + f'.{args.worker}')
