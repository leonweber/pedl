import re
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

from pedl.utils import SegtokSentenceSplitter, get_pmid, chunks, cache_root, replace_consistently_dict
from pedl.data_getter import DataGetterAPI

INDEX_NAME = "pubtator_masked"


def _add_masked_entities(elastic_doc, mask_types=None) -> None:
    elastic_doc["entities_masked"] = {}
    entities = list(elastic_doc["entities"])

    span_to_replacement = {}

    idx_mask = 1
    for entity in entities:
        spans = elastic_doc["entities"][entity]

        entity_type = entity.split("|")[1]
        replacement = None
        if entity_type in mask_types:
            for span in spans:
                span = tuple(span)
                if span in span_to_replacement:
                    continue
                else:
                    if replacement is None:
                        replacement = "<" + mask_types[entity_type] + str(idx_mask) + "/>"
                        idx_mask += 1
                    # fix spans crossing sentence boundaries
                    if span[1] > len(elastic_doc["text"]):
                        span = (span[0], len(elastic_doc["text"]))

                    span_to_replacement[span] = replacement

    text = elastic_doc["text"]
    text_masked, old_span_to_new = replace_consistently_dict(text, span_to_replacement)

    elastic_doc["text_masked"] = text_masked

    for entity in entities:
        spans = elastic_doc["entities"][entity]
        new_spans = []
        for span in spans:
            span = tuple(span)
            if span in old_span_to_new:
                new_spans.append(old_span_to_new[span])
            else:
                new_spans.append(span)
        elastic_doc["entities_masked"][entity] = new_spans



    # assert len(elastic_doc["entities_masked"]) == len(elastic_doc["entities"])
    # for entity in elastic_doc["entities_masked"]:
    #     assert len(elastic_doc["entities_masked"][entity]) >= len(elastic_doc["entities"][entity])


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
                        entities_ann = DataGetterAPI.get_entities_from_annotation(annotation=annotation,
                                                                                  homologue_mapping={})
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
                            if entity[0][0] - sentence.start_pos < 0:
                                if len(entities_passage) == 0:
                                    entity = None
                                    break
                                else:
                                    entity = entities_passage.pop()
                                continue

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


def build_index(pubtator_file, n_processes, elasticsearch, masked_types=None, entity_marker: dict = None):

    n_processes = n_processes or mp.cpu_count()
    if elasticsearch.server.startswith("https://"):
        scheme, host, port = elasticsearch.server.split(":")
        host = host[2:]
    else:
        host, port = elasticsearch.server.split(":")

    if not elasticsearch.password and not elasticsearch.ca_certs:
        client = Elasticsearch(hosts=[{"host": host,
                                            "port": int(port),
                                            "scheme": "http",
                                            }], timeout=3000,
                                    )
    elif not elasticsearch.password:
        client = Elasticsearch(hosts=[{"host": host,
                                            "port": int(port),
                                            "scheme": "https",
                                            }], timeout=3000,
                                    ca_certs=elasticsearch.ca_certs
                                    )

    else:
        client = Elasticsearch(hosts=[{"host": host,
                                            "port": int(port),
                                            "scheme": "http",
                                            }], timeout=3000,
                                    basic_auth=(elasticsearch.username, elasticsearch.password),
                                    ca_certs=elasticsearch.ca_certs
                                    )


    if client.indices.exists(index=INDEX_NAME):
        client.indices.delete(index=INDEX_NAME)

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
    files = list(pubtator_file.glob("*BioC.XML"))
    processes = []
    for file_chunk in chunks(files, len(files) // n_processes - 1):
        p = ctx.Process(
            target=_process_pubtator_files, args=(file_chunk, q, masked_types, entity_marker)
        )
        p.start()
        processes.append(p)

    pbar = tqdm(desc="Building pubtator elastic index", total=len(files))
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

