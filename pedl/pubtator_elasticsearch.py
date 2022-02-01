import sys
from collections import defaultdict
from operator import itemgetter
from pathlib import Path
from typing import List
import multiprocessing as mp

import bioc
import numpy as np
from elasticsearch import Elasticsearch, helpers
from tqdm import tqdm

from pedl.utils import SegtokSentenceSplitter, get_pmid, chunks, cache_root, replace_consistently
from pedl.data_getter import DataGetterAPI

INDEX_NAME = "pubtator_masked"


def _add_masked_entities(elastic_doc, mask_types=None) -> None:
    elastic_doc["entities_masked"] = {}
    entities = list(elastic_doc["entities"])

    offsets = []
    lengths_orig = []
    entity_to_offset_idx = defaultdict(list)

    for entity in entities:
        spans = elastic_doc["entities"][entity]
        for span in spans:
            entity_to_offset_idx[entity].append(len(offsets))
            offsets.append(span[0])
            lengths_orig.append(span[1] - span[0])

    idx_mask = 1
    text = elastic_doc["text"]
    lengths_new = np.array(lengths_orig)
    offsets_new = np.array(offsets)
    for entity in entities:
        entity_type = entity.split("|")[1]
        if entity_type in mask_types:
            replacement = "<" + mask_types[entity_type] + str(idx_mask) + "/>"
            for offset_idx in entity_to_offset_idx[entity]:
                text, offsets_new = replace_consistently(offset=offsets_new[offset_idx],
                                     length=lengths_orig[offset_idx],
                                     replacement=replacement,
                                     text=text,
                                     offsets=offsets_new)
                lengths_new[offset_idx] = len(replacement)
            idx_mask += 1

    elastic_doc["text_masked"] = text

    for entity, offset_indices in entity_to_offset_idx.items():
        spans_new = []
        for offset_idx in offset_indices:
            start = offsets_new[offset_idx]
            end = start + lengths_new[offset_idx]
            spans_new.append([start, end])
        elastic_doc["entities_masked"][entity] = spans_new


def _process_pubtator_files(files: List[Path], q: mp.Queue, mask_types=None, entity_marker: dict = None):
    sentence_splitter = SegtokSentenceSplitter(entity_marker=entity_marker)
    for file in files:
        actions = []
        with file.open() as f:
            collection = bioc.load(f)
            for i, document in enumerate(collection.documents):
                pmid, is_fulltext = get_pmid(document)

                for passage in document.passages:
                    if is_fulltext and (
                            passage.infons["type"] == "ref" or # this is a reference
                            passage.infons.get("section_type", "").lower() in {"title", "abstract"}): # this was already added via the abstract
                        continue

                    entities_passage = []
                    for annotation in passage.annotations:
                        entities_ann = DataGetterAPI.get_entities_from_annotation(annotation,
                                                                                  {})
                        for entity in entities_ann:
                            for location in annotation.locations:
                                span = (location.offset-passage.offset,
                                        location.end-passage.offset)
                                entities_passage.append((span, entity))
                    entities_passage = sorted(entities_passage, key=itemgetter(0), reverse=True) # reversed because we pop from end, which gives us O(1)

                    if not entities_passage:
                        continue

                    entity = entities_passage.pop()
                    for sentence in sentence_splitter.split(passage.text):
                        if not entity: # we have exhausted all entities
                            break

                        if entity[0][0] > sentence.end_pos: # this sentence contains no entity => skip
                            continue
                        elastic_doc = {
                            "text": sentence.text,
                            "cellline": [],
                            "chemical": [],
                            "disease": [],
                            "gene": [],
                            "proteinmutation": [],
                            "dnamutation": [],
                            "snpmutation": [],
                            "species": [],
                            "entities": defaultdict(list),
                            "pmid": pmid,
                            "span": [sentence.start_pos + passage.offset, sentence.end_pos + passage.offset + passage.offset + passage.offset]
                        }
                        while entity[0][0] < sentence.end_pos:
                            if entity[1].type.lower() in elastic_doc:
                                elastic_doc[entity[1].type.lower()].append(entity[1].cuid)
                                span = [entity[0][0] - sentence.start_pos, entity[0][1] - sentence.start_pos]
                                elastic_doc["entities"][entity[1].cuid + "|" + entity[1].type].append(span)
                            if len(entities_passage) == 0:
                                entity = None
                                break
                            else:
                                entity = entities_passage.pop()

                        _add_masked_entities(elastic_doc, mask_types)

                        action = {
                            "_index": INDEX_NAME,
                            "_source": elastic_doc
                        }
                        actions.append(action)

        q.put(actions)


def build_index(pubtator_path, n_processes, masked_types=None, entity_marker: dict = None):

    n_processes = n_processes or mp.cpu_count()
    client = Elasticsearch(timeout=3000)

    if client.indices.exists(INDEX_NAME):
        client.indices.delete(INDEX_NAME)

    client.indices.create(
        index=INDEX_NAME,
        mappings={
            "dynamic": False,
            "properties": {
                "cellline": {"type": "keyword"},
                "chemical": {"type": "keyword"},
                "disease": {"type": "keyword"},
                "gene": {"type": "keyword"},
                "proteinmutation": {"type": "keyword"},
                "dnamutation": {"type": "keyword"},
                "snpmutation": {"type": "keyword"},
                "species": {"type": "keyword"},
                "pmid": {"type": "keyword"},
            }
        }
                          )

    ctx = mp.get_context()
    q = ctx.Queue(maxsize=10)
    files = list(pubtator_path.glob("*bioc.xml"))
    processes = []
    for file_chunk in chunks(files, len(files) // n_processes - 1):
        p = ctx.Process(
            target=_process_pubtator_files, args=(file_chunk, q, masked_types, entity_marker)
        )
        p.start()
        processes.append(p)

    pbar = tqdm(desc="Building pubtator elasticsearch index", total=len(files))
    n_files_processed = 0
    while n_files_processed < len(files):
        elastic_docs = q.get()
        num_attempts = 0
        all_errors = []
        while errors := helpers.bulk(client, elastic_docs)[1]:
            num_attempts += 1
            all_errors += errors
            if num_attempts >= 30:
                print("Too many attempts. Errors:")
                print(all_errors)
                sys.exit(1)
        n_files_processed += 1
        pbar.update()

    for p in processes:
        p.join()
