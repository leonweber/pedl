import abc
import os
import logging
import warnings
from collections import defaultdict
from typing import Optional, Set, List, Dict

import bioc
import diskcache
import numpy as np
import requests
from elasticsearch import Elasticsearch
from lxml import etree
from omegaconf import OmegaConf
from tqdm import tqdm

from pedl.utils import get_homologue_mapping, cache_root, SegtokSentenceSplitter, \
    cached_path, unpack_file, chunks, Entity, Sentence, get_pmid, \
    replace_consistently_dict, doc_synced



class DataGetter(abc.ABC):

    @abc.abstractmethod
    def get_sentences(self, head: Entity, tail: Entity) -> List[Sentence]:
        pass


class DataGetterPubtator(DataGetter):

    def __init__(self, elasticsearch, entity_marker: dict = None, entity_to_mask: dict = None,
                 max_size: int = 1000):
        if elasticsearch.server.startswith("https://"):
            scheme, host, port = elasticsearch.server.split(":")
            host = host[2:]
        else:
            host, port = elasticsearch.server.split(":")
        self.types_to_blind = {"gene"}

        if not elasticsearch.password and not elasticsearch.ca_certs:
            self.client = Elasticsearch(hosts=[{"host": host,
                                                "port": int(port),
                                                "scheme": "http",
                                                }], timeout=3000,
                                        )
        elif not elasticsearch.password:
            self.client = Elasticsearch(hosts=[{"host": host,
                                                "port": int(port),
                                                "scheme": "https",
                                                }], timeout=3000,
                                        ca_certs=elasticsearch.ca_certs
                                        )

        else:
            self.client = Elasticsearch(hosts=[{"host": host,
                                                "port": int(port),
                                                "scheme": "http",
                                                }], timeout=3000,
                                        basic_auth=(elasticsearch.username, elasticsearch.password),
                                        ca_certs=elasticsearch.ca_certs
                                        )

        if entity_marker:
            self.et = entity_marker
        else:
            self.et = {"head_start": '<e1>',
                       "head_end": '</e1>',
                       "tail_start": '<e2>',
                       "tail_end": '</e2>'}

        self.entity_to_mask = entity_to_mask
        self.max_size = max_size

    def get_sentences(self, head: Entity, tail: Entity):
        processed_sentences = []
        query = {
            "bool": {
                "must": [
                    {"term": {head.type.lower(): head.cuid}},
                    {"term": {tail.type.lower(): tail.cuid}},
                ]
            }
        }
        result = self.client.search(query=query, index="pubtator_masked", size=self.max_size)
        retrieved_sentences = result["hits"]["hits"]
        for sentence in retrieved_sentences:
            sentence = sentence["_source"]

            spans_head = sentence["entities"][str(head)]
            spans_tail = sentence["entities"][str(tail)]
            spans_head_masked = sentence["entities_masked"][str(head)]
            spans_tail_masked = sentence["entities_masked"][str(tail)]

            text_orig = sentence["text"]
            text_orig_masked = sentence["text_masked"]

            for idx_head, span_head in enumerate(spans_head):
                for idx_tail, span_tail in enumerate(spans_tail):
                    span_to_replacement = {
                        tuple(span_head): self.et['head_start'] + text_orig[span_head[0]:span_head[-1]]
                        + self.et['head_end'],
                        tuple(span_tail): self.et['tail_start'] + text_orig[span_tail[0]:span_tail[-1]]
                        + self.et['tail_end'],
                    }
                    text, _ = replace_consistently_dict(text=text_orig, span_to_replacement=span_to_replacement)

                    span_head_masked = spans_head_masked[idx_head]
                    span_tail_masked = spans_tail_masked[idx_tail]

                    span_to_replacement = {
                        tuple(
                            span_head_masked): self.et['head_start'] + text_orig_masked[span_head_masked[0]:span_head_masked[1]] + self.et['head_end'],
                        tuple(
                            span_tail_masked): self.et['tail_start'] + text_orig_masked[span_tail_masked[0]:span_tail_masked[1]] + self.et['tail_end'],
                    }
                    text_masked, _ = replace_consistently_dict(text=text_orig_masked,
                                                            span_to_replacement=span_to_replacement)

                    processed_sentences.append(
                        Sentence(text=text,
                                 text_blinded=text_masked,
                                 pmid=sentence["pmid"],
                                 start_pos=0,
                                 entity_marker=self.et
                                 )
                    )

        return processed_sentences


class DataGetterAPI(DataGetter):
    CHUNK_SIZE = 100

    def __init__(
            self,
            gene_universe: Optional[Set[str]] = None,
            chemical_universe: Optional[Set[str]] = None,
            expand_species: Optional[List[str]] = None,
            entity_to_mask: Optional[Dict[str, str]] = None,
            entity_marker: dict = None
    ):
        self.gene_universe = gene_universe or set()
        self.chemical_universe = chemical_universe or set()
        self.expand_species = expand_species or []
        self.entity_to_mask = entity_to_mask or set()
        if self.expand_species:
            self.homologue_mapping = get_homologue_mapping(
                self.expand_species, self.gene_universe
            )
        else:
            self.homologue_mapping = {}
        self.gene2pmid = self.get_gene2pmid()
        self.chemical2pmid = self.get_chemical2pmid()
        self._document_cache = diskcache.Cache(
            directory=str(cache_root / "document_cache"),
            eviction_policy="least-recently-used",
        )
        if entity_marker:
            self.entity_marker = entity_marker
        else:
            self.entity_marker = {"head_start": '<e1>',
                                  "head_end": '</e1>',
                                  "tail_start": '<e2>',
                                  "tail_end": '</e2>'}
        self.sentence_splitter = SegtokSentenceSplitter(self.entity_marker)

    def get_gene2pmid(self):
        gene2pmid = defaultdict(set)

        final_path = cache_root / "data" / "gene2pubtatorcentral"
        if not final_path.exists():
            print("Downloading gene2pubtatorcentral...")
            path = cached_path(
                "https://ftp.ncbi.nlm.nih.gov/pub/lu/PubTatorCentral/gene2pubtatorcentral.gz",
                "data",
            )
            unpack_file(path, final_path)
        doc_synced(final_path, 'gene')

        with final_path.open() as f:
            for line in tqdm(f, total=70000000, desc="Loading gene2pubtatorcentral"):
                line = line.strip()
                fields = line.split("\t")
                gene_id = fields[2]
                pmid = fields[0]
                normalizers = fields[4]
                if "GNormPlus" not in normalizers:
                    continue

                if gene_id in self.gene_universe:
                    gene2pmid[gene_id].add(pmid)
                elif gene_id in self.homologue_mapping:
                    for mapped_gene_id in self.homologue_mapping[
                        gene_id
                    ]:  # mapped_gene_id is from self.protein_universe
                        gene2pmid[mapped_gene_id].add(pmid)

        return dict(gene2pmid)

    def get_chemical2pmid(self):
        chemical2pmid = defaultdict(set)

        if not self.chemical_universe:
            return chemical2pmid

        final_path = cache_root / "data" / "chemical2pubtatorcentral"
        if not final_path.exists():
            print("Downloading chemical2pubtatorcentral...")
            path = cached_path(
                "https://ftp.ncbi.nlm.nih.gov/pub/lu/PubTatorCentral/chemical2pubtatorcentral.gz",
                "data",
            )
            unpack_file(path, final_path)

        doc_synced(final_path, 'chemical')
        with final_path.open() as f:
            for line in tqdm(
                    f, total=104567794, desc="Loading chemical2pubtatorcentral"
            ):
                line = line.strip()
                fields = line.split("\t")
                chemical_id = fields[2]
                pmid = fields[0]
                normalizers = fields[4]
                if "TaggerOne" not in normalizers:
                    continue

                if chemical_id in self.chemical_universe:
                    chemical2pmid[chemical_id].add(pmid)
        return dict(chemical2pmid)

    def maybe_map_to_pmcid(self, pmids):
        pmid_to_pmcid = {}

        service_root = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
        for pmid_chunk in chunks(pmids, 200):
            ids = ",".join(pmid_chunk)
            response = requests.get(service_root, params={"ids": ids})
            try:
                for record in etree.fromstring(response.content).xpath("//record"):
                    if "pmcid" in record.attrib:
                        pmid_to_pmcid[record.attrib["pmid"]] = record.attrib["pmcid"]
            except etree.XMLSyntaxError:
                warnings.warn(
                    "Failed to parse PubTator response."
                    " You are probably issuing too many requests."
                    " Please use a local copy of PubTator or disable api_fallback, if you already do."
                )

        return pmid_to_pmcid

    @staticmethod
    def get_entities_from_annotation(annotation: bioc.BioCAnnotation, homologue_mapping: Dict) -> Set[Entity]:
        """
        Extract entities from `annotation` and expand them with homologues from
        `self.expand_species` if it is a Gene.
        """
        if "identifier" in annotation.infons:
            identifiers = annotation.infons["identifier"]
            if not identifiers:
                identifiers = ""
        elif "Identifier" in annotation.infons:
            identifiers = annotation.infons["Identifier"]
            if not identifiers:
                identifiers = ""
        else:
            identifiers = ""

        identifiers = set(identifiers.split(";"))

        expanded_identifiers = identifiers.copy()
        if annotation.infons["type"] == "Gene":
            for cuid in identifiers:
                expanded_identifiers.update(homologue_mapping.get(cuid, {}))

        return set(Entity(cuid=cuid, type=annotation.infons["type"]) for cuid in expanded_identifiers)

    def get_sentences_from_document(self, entity1: Entity, entity2: Entity, document):
        sentences = []
        for passage in document.passages:
            entity1_locations = []
            entity1_lengths = []
            entity2_locations = []
            entity2_lengths = []
            for annotation in passage.annotations:
                if annotation.infons["type"] not in {entity1.type, entity2.type}:
                    continue

                entities_ann = self.get_entities_from_annotation(annotation, self.homologue_mapping)

                if entity1 in entities_ann:
                    for loc in annotation.locations:
                        entity1_locations.append(loc.offset - passage.offset)
                        entity1_lengths.append(loc.length)
                if entity2 in entities_ann:
                    for loc in annotation.locations:
                        entity2_locations.append(loc.offset - passage.offset)
                        entity2_lengths.append(loc.length)

            entity1_locations_arr = np.array(entity1_locations).reshape(-1, 1)
            entity2_locations_arr = np.array(entity2_locations).reshape(1, -1)
            dists = abs(entity1_locations_arr - entity2_locations_arr)
            for i, j in zip(*np.where(dists <= 300)):
                loc_ent1 = entity1_locations_arr[i, 0]
                loc_ent2 = entity2_locations_arr[0, j]
                len_ent1 = entity1_lengths[i]
                len_ent2 = entity2_lengths[j]
                sentence = self.get_sentence(
                    passage=passage,
                    offset_ent1=loc_ent1,
                    offset_ent2=loc_ent2,
                    len_ent1=len_ent1,
                    len_ent2=len_ent2,
                    pmid=get_pmid(document)[0],
                )
                if sentence:
                    sentences.append(sentence)

        return sentences

    def get_pmids(self, entity: Entity) -> Set[str]:
        if entity.type == "Chemical":
            cuid_to_pmid = self.chemical2pmid
        elif entity.type == "Gene":
            cuid_to_pmid = self.gene2pmid
        else:
            raise ValueError

        return cuid_to_pmid.get(entity.cuid, set())

    def get_sentences(self, head: Entity, tail: Entity) -> List[Sentence]:
        sentences = []
        pmids = sorted(self.get_pmids(head) & self.get_pmids(tail))
        if not pmids:
            return []

        for documents in self.get_documents_from_api(pmids):
            for document in documents:
                sentences += self.get_sentences_from_document(
                    entity1=head, entity2=tail, document=document
                )

        return sentences

    def cache_documents(self, documents: List[bioc.BioCDocument]) -> None:
        logging.info(f"Caching {len(documents)} documents")
        for document in documents:
            pmid = get_pmid(document)[0]
            self._document_cache[pmid] = document

    def get_documents_from_api(self, pmids):
        service_root = "https://www.ncbi.nlm.nih.gov/research/pubtator-api/publications/export/biocxml"
        pmids = list(pmids)

        if len(pmids) > self.CHUNK_SIZE:
            pbar = tqdm(desc="Collecting", total=len(pmids))
        else:
            pbar = None

        cached_pmids = [i for i in pmids if i in self._document_cache]

        for pmid_chunk in chunks(cached_pmids, self.CHUNK_SIZE):
            if pbar:
                pbar.update(len(pmid_chunk))
            yield [self._document_cache[i] for i in pmid_chunk]

        uncached_pmids = [i for i in pmids if i not in cached_pmids]
        pmid_to_pmcid = self.maybe_map_to_pmcid(uncached_pmids)

        pmids_to_retreive = [i for i in uncached_pmids if i not in pmid_to_pmcid]
        pmcids_to_retreive = [
            pmid_to_pmcid[i] for i in uncached_pmids if i in pmid_to_pmcid
        ]

        pmcid_to_pmid = {v: k for k, v in pmid_to_pmcid.items()}

        for pmid_chunk in list(chunks(pmids_to_retreive, self.CHUNK_SIZE)):

            result = requests.get(
                service_root, params={"pmids": ",".join(pmid_chunk), "concepts": "gene,chemical"}
            )
            collection = bioc.loads(result.content.decode())
            yield collection.documents
            if pbar:
                pbar.update(len(pmid_chunk))
            self.cache_documents(collection.documents)

        for pmcid_chunk in list(chunks(pmcids_to_retreive, self.CHUNK_SIZE)):

            result = requests.get(
                service_root,
                params={"pmcids": ",".join(pmcid_chunk), "concepts": "gene"},
            )
            collection = bioc.loads(result.content.decode())
            yield collection.documents
            if pbar:
                pbar.update(len(pmcid_chunk))
            self.cache_documents(collection.documents)

    def get_sentence(
            self,
            passage,
            offset_ent1,
            offset_ent2,
            len_ent1,
            len_ent2,
            pmid,
            allow_multi_sentence=False,
    ):

        if offset_ent1 < offset_ent2:
            left_start = offset_ent1
            right_start = offset_ent2
        else:
            left_start = offset_ent2
            right_start = offset_ent1
        sents = self.sentence_splitter.split(passage.text.strip())
        snippet_start = None
        snippet_end = None
        for sent in sents:
            if sent.end_pos >= left_start >= sent.start_pos:
                snippet_start = sent.start_pos

            if (
                    sent.end_pos >= right_start >= sent.start_pos
            ):  # is sentence after right entity
                snippet_end = sent.end_pos

        if snippet_start is not None and snippet_end is None:
            snippet_end = sent.end_pos
        if snippet_start is None:
            return None

        if not allow_multi_sentence and not any(
                snippet_start == i.start_pos and snippet_end == i.end_pos for i in sents
        ):
            return None  # is multi sentence

        text = passage.text[snippet_start:snippet_end]
        span_to_replacement = {}
        span_to_replacement_masked = {}
        masked_count = 0
        entity_to_spans = defaultdict(list)
        for ann in passage.annotations:
            for loc in ann.locations:
                if snippet_end >= loc.offset - passage.offset >= snippet_start:
                    entities = self.get_entities_from_annotation(ann, self.homologue_mapping)

                    start = loc.offset - passage.offset - snippet_start
                    end = loc.end - passage.offset - snippet_start
                    for entity in entities:
                        entity_to_spans[entity].append((start, end))

        for entity, spans in entity_to_spans.items():
            masked_count += 1
            for start, end in spans:
                ent_text = text[start:end]
                if entity.type in self.entity_to_mask:
                    replacement_masked = f"<{self.entity_to_mask[entity.type]}{masked_count}/>"
                else:
                    replacement_masked = ent_text
                replacement = ent_text

                if start == offset_ent1 - snippet_start:
                    replacement = f"{self.entity_marker['head_start']}{replacement}{self.entity_marker['head_end']}"
                    replacement_masked = f"{self.entity_marker['head_start']}{replacement_masked}{self.entity_marker['head_end']}"

                if start == offset_ent2 - snippet_start:
                    replacement = f"{self.entity_marker['tail_start']}{replacement}{self.entity_marker['tail_end']}"
                    replacement_masked = f"{self.entity_marker['tail_start']}{replacement_masked}{self.entity_marker['tail_end']}"
                span_to_replacement[(start, end)] = replacement
                span_to_replacement_masked[(start, end)] = replacement_masked

        text_marked, _ = replace_consistently_dict(text, span_to_replacement)
        text_masked, _ = replace_consistently_dict(text, span_to_replacement_masked)

        if self.entity_marker['head_start'] not in text_marked or self.entity_marker['head_end'] not in text_marked:
            return None # Weird encoding error

        return Sentence(pmid=pmid, text=text_marked, text_blinded=text_masked, start_pos=snippet_start, entity_marker=self.entity_marker)
