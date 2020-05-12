import os
import json
from pathlib import Path
from collections import defaultdict
from argparse import ArgumentParser

from pairs import PairGetter

from gen_ann_file import load_annotations, get_augmented_offset_lines
from tax_ids import TAX_IDS

TypedEntity = PairGetter.TypedEntity


def triple_to_examples(triple, pair_getter: PairGetter, doc):
    e1, r, e2, pmids = triple
    sentences = pair_getter.get_sentences((e1, e2), doc)
    examples = set()
    for sent_idx, sentence in enumerate(sentences):
        e1_span = sentence.spans[e1]
        e2_span = sentence.spans[e2]

        if e1_span[0] < e2_span[0]:
            left_span = e1_span
            right_span = e2_span
            first_ent = 'e1'
        else:
            left_span = e2_span
            right_span = e1_span
            first_ent = 'e2'

        e1_text = " ".join(sentence.tokens[e1_span[0]:e1_span[1]])
        e2_text = " ".join(sentence.tokens[e2_span[0]:e2_span[1]])

        left_context = " ".join(sentence.tokens[:left_span[0]])
        mid_context = " ".join(sentence.tokens[left_span[1]:right_span[0]])
        right_context = " ".join(sentence.tokens[right_span[1]:])

        if first_ent == "e1":
            tokens = f"{left_context} <e1>{e1_text}</e1> {mid_context} <e2>{e2_text}</e2> {right_context}"
        else:
            tokens = f"{left_context} <e2>{e2_text}</e2> {mid_context} <e1>{e1_text}</e1> {right_context}"

        is_evidence = sentence.pmid in pmids
        supervision_type = "direct" if is_evidence else "distant"

        examples.add("\t".join([e1.id, e2.id, e1_text, e2_text, r, tokens, supervision_type, doc.pmid]))

    return examples


def write_examples(getter: PairGetter, offset_lines, fname, train_pairs, dev_pairs, test_pairs, suffix):
    with open(str(dataset_dir / "train.txt") + suffix, "w") as f_train, open(str(dataset_dir / "dev.txt") + suffix,
                                                                             "w") as f_dev, \
            open(str(dataset_dir / "test.txt") + suffix, "w") as f_test:
        for i, doc in enumerate(getter.get_relevant_docs(offset_lines)):

            if args.test and i % 100 == 0:
                print(f"TEST RUN: {i}/1000 ")
            if args.test and i > 1000:
                break

            for pair in getter._pmid_to_entity_sets[doc.pmid]:
                if pair in train_pairs:
                    for triple in train_pairs[pair]:
                        examples = "\n".join(triple_to_examples(triple, getter, doc))
                        if examples:
                            f_train.write(examples)
                            f_train.write("\n")
                elif pair in dev_pairs:
                    for triple in dev_pairs[pair]:
                        examples = "\n".join(triple_to_examples(triple, getter, doc))
                        if examples:
                            f_dev.write(examples)
                            f_dev.write("\n")
                elif pair in test_pairs:
                    for triple in test_pairs[pair]:
                        examples = "\n".join(triple_to_examples(triple, getter, doc))
                        if examples:
                            f_test.write("\n".join(triple_to_examples(triple, getter, doc)) + "\n")
                            f_test.write("\n")
                else:
                    raise ValueError(f"Pair {pair} is in neither train/dev/test")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('--pmc_dir', default=Path('data/pmc_bioc'), type=Path)
    parser.add_argument('--worker', default='0', type=int)
    parser.add_argument('--n_workers', default='1', type=int)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--pmid_blacklist', default=None)
    parser.add_argument('--mapping', default=None)
    parser.add_argument('--species', default="")

    args = parser.parse_args()

    processed_pmids = set()
    if args.pmid_blacklist:
        with open(args.pmid_blacklist) as f:
            for line in f:
                processed_pmids.add(line.strip())

    pairs = set()
    with open(args.input + '.train.json') as f:
        triples = json.load(f)
        train_triples = []
        train_pairs = defaultdict(list)
        items = list(triples.items())
        for triple, pmids in items:
            triple = tuple(triple.split(",") + [pmids])
            e1 = TypedEntity(triple[0], 'Gene')
            e2 = TypedEntity(triple[2], 'Gene')
            triple = (e1, triple[1], e2, triple[3])
            train_triples.append(triple)
            train_pairs[(triple[0], triple[2])].append(triple)

    with open(args.input + '.dev.json') as f:
        triples = json.load(f)
        dev_triples = []
        dev_pairs = defaultdict(list)
        items = list(triples.items())
        for triple, pmids in items:
            triple = tuple(triple.split(",") + [pmids])
            e1 = TypedEntity(triple[0], 'Gene')
            e2 = TypedEntity(triple[2], 'Gene')
            triple = (e1, triple[1], e2, triple[3])
            dev_triples.append(triple)
            dev_pairs[(triple[0], triple[2])].append(triple)

    with open(args.input + '.test.json') as f:
        triples = json.load(f)
        test_triples = []
        test_pairs = defaultdict(list)
        items = list(triples.items())
        for triple, pmids in items:
            triple = tuple(triple.split(",") + [pmids])
            e1 = TypedEntity(triple[0], 'Gene')
            e2 = TypedEntity(triple[2], 'Gene')
            triple = (e1, triple[1], e2, triple[3])
            test_triples.append(triple)
            test_pairs[(triple[0], triple[2])].append(triple)

    pairs = set(train_pairs) | set(dev_pairs) | set(test_pairs)

    anns_path = 'bioconcepts2pubtatorcentral.offset'

    species = [TAX_IDS[s] for s in args.species.split(',')]

    mapping = {}
    if args.mapping:
        with open(args.mapping) as f:
            mapping['Gene'] = json.load(f)

    with open(anns_path) as f:
        offset_lines = get_augmented_offset_lines(f, pmc_dir=args.pmc_dir, types={'Gene'}, mapping=mapping,
                                                  homologue_species=species,
                                                  test=args.test, worker=args.worker, n_workers=args.n_workers)
        anns = load_annotations(offset_lines)
        getter = PairGetter(entity_sets=pairs, anns=anns)

    with open(anns_path) as f:
        offset_lines = get_augmented_offset_lines(f, pmc_dir=args.pmc_dir, types={'Gene'}, mapping=mapping,
                                                  homologue_species=species,
                                                  test=args.test, worker=args.worker, n_workers=args.n_workers,
                                                  relevant_pmids=getter.relevant_pmids)

        dataset_dir = Path(str(args.input) + "_raw")
        os.makedirs(dataset_dir, exist_ok=True)

        suffix = f".{args.worker}.{args.species}"

        write_examples(getter=getter,
                       offset_lines=offset_lines,
                       fname=args.input, train_pairs=train_pairs,
                       dev_pairs=dev_pairs, test_pairs=test_pairs, suffix=suffix)
