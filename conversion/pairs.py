import itertools
import logging
import string
from bisect import bisect_right, bisect_left
from collections import defaultdict, namedtuple
from dataclasses import dataclass
from itertools import takewhile
from typing import List, Optional, Set, Dict, Any, Tuple
import pickle

import spacy
from tqdm import tqdm
import numpy as np

import scispacy


@dataclass
class Annotation:
    start: int
    end: int
    mention: str
    type: str
    id: Optional[str]

    @classmethod
    def from_string(cls, string: str):
        fields = string.split("\t")
        if len(fields) > 5:
            id_ = fields[5]
        else:
            id_ = None
        return Annotation(start=int(fields[1]), end=int(fields[2]), mention=fields[3], type=fields[4], id=id_)


@dataclass
class Sentence:
    tokens: List[str]
    spans: Dict[str, Tuple[int, int]]
    pmid: str
    doc_spans: Dict[str, Tuple[int, int]] = None


class Document:
    annotations: List[Annotation]
    nlp = spacy.load("en_core_sci_sm", disable=['tagger', 'ner'], max_length=5000000)

    def __init__(self, pmid, tokens, annotations):
        self.pmid = pmid
        self.annotations = annotations
        self.tokens = tokens
        try:
            self.spacy_doc = self.nlp(" ".join(tokens))
        except ValueError:
            self.spacy_doc = None

    @classmethod
    def from_string(cls, lines: List[str]):
        pmid = lines[0].split('|')[0]
        texts = []
        annotations = []
        for line in lines:
            text_split = line.split("|")
            if len(text_split) > 1 and text_split[1] in {'a', 't', 's'}:
                texts.append(text_split[2])
            else:
                annotations.append(Annotation.from_string(line))

        text = " ".join(texts)
        return Document(pmid=pmid, tokens=text.split(" "), annotations=annotations)


def get_span(start, end, token_starts):
    """
    Adapt annotations to token spans
    """
    
    #token_ends = [len(tokens[0])]
    #for token in tokens[1:]:
        #new_end = token_ends[-1] + len(token) + 1
        #token_ends.append(new_end)

    new_start = bisect_right(token_starts, start) - 1
    new_end = bisect_left(token_starts, end)

    return (new_start, new_end)



class PairGetter:

    TypedEntity = namedtuple('TypedEntity', 'id type')

    def __init__(self, entity_sets, anns):
        self.anns = anns
        self.entity_sets = entity_sets
        self._entity_sets_to_pmids = self.get_entity_sets_to_docs(entity_sets)
        self._pmid_to_entity_sets = defaultdict(set)
        for entity_set, docs in self._entity_sets_to_pmids.items():
            for doc in docs:
                self._pmid_to_entity_sets[doc].add(entity_set)
        
    def get_entity_sets_to_docs(self, entity_sets):
        print("Generating pair to document mapping")
        entity_to_doc = defaultdict(set)
        entity_sets_to_doc = {}

        for pmid, doc_anns in tqdm(self.anns.items()):
            for ann in doc_anns:
                entity_to_doc[self.TypedEntity(ann.id, ann.type)].add(pmid)

        for entity_set in tqdm(entity_sets):
            entity_sets_to_doc[entity_set] = entity_to_doc[entity_set[0]]
            for entity in entity_set[1:]:
                # cannot use &= here, because it will alter entity_to_doc[entity_set[0]] and lead to many hours of debugging :(
                entity_sets_to_doc[entity_set] = entity_to_doc[entity] & entity_sets_to_doc[entity_set]

        return entity_sets_to_doc

    @property
    def relevant_pmids(self):
        return set(self._pmid_to_entity_sets)

    def get_relevant_docs(self, offset_lines) -> Dict[str, Document]:
        print("Extracting documents")
        doc_lines = []
        active_pmid = None
        for line in offset_lines:
            if not line.strip():
                continue

            pmid = "".join(takewhile(lambda x: x in string.digits, line))

            if active_pmid != pmid and doc_lines:
                yield Document.from_string(doc_lines)
                doc_lines = []

            if pmid in self._pmid_to_entity_sets:
                doc_lines.append(line.strip())
                active_pmid = pmid
        if doc_lines:
            yield Document.from_string(doc_lines)


    def get_mentions(self, entity, doc, context_size=250):
        sentences: List[Sentence] = []
        token_starts = [0]
        for token in doc.tokens[:-1]:
            new_start = token_starts[-1] + len(token) + 1
            token_starts.append(new_start)
        entity_mentions = [a for a in doc.annotations if a.id == entity.id and a.type == entity.type]

        for mention in entity_mentions:
            span = get_span(mention.start, mention.end, token_starts)
            left_boundary = max(span[0] - context_size, 0)
            right_boundary = min(span[1] + context_size, len(doc.tokens))
            tokens = doc.tokens[left_boundary:right_boundary]
            new_span = (span[0]-left_boundary, span[1]-left_boundary)
            spans = {entity: new_span}
            doc_spans = {entity: span}
            sentences.append(Sentence(tokens,
                                        spans=spans,
                                        doc_spans=doc_spans,
                                        pmid=doc.pmid
                                        ))
        
        return sentences



    def get_sentences(self, entity_set, doc, max_char_dist=300):

        if len(entity_set) != 2:
            raise NotImplementedError # need to adapt from gene pairs to entity sets

        e1 = entity_set[0]
        e2 = entity_set[1]

        sentences: List[Sentence] = []

        e1_mentions = [a for a in doc.annotations if a.id == e1.id and a.type == e1.type]
        e2_mentions = [a for a in doc.annotations if a.id == e2.id and a.type == e2.type]

        for e1_mention in e1_mentions:
            for e2_mention in e2_mentions:
                if e1_mention.start < e2_mention.start:
                    left_ent = e1_mention
                    right_ent = e2_mention
                else:
                    left_ent = e2_mention
                    right_ent = e1_mention

                if doc.spacy_doc and abs(e1_mention.start - e2_mention.start) <= max_char_dist:
                    snippet_start = None
                    snippet_end = None
                    for sent in doc.spacy_doc.sents:
                        sent_start = sent[0].idx
                        sent_end = sent[-1].idx + len(str(sent[-1]))
                        if sent_end >= left_ent.start >= sent_start:
                            snippet_start = sent_start

                        if sent_end >= right_ent.end >= sent_start: # is sentence after right entity
                            snippet_end = sent_end

                    if not (snippet_start and snippet_end):
                        continue

                    # adapt entity positions to text snippet
                    e1_start = e1_mention.start - snippet_start
                    e1_end = e1_mention.end - snippet_start
                    e2_start = e2_mention.start - snippet_start
                    e2_end = e2_mention.end - snippet_start
                    text = str(doc.spacy_doc)[snippet_start:snippet_end]
                    tokens = text.split()

                    token_starts = [0]
                    for token in tokens[:-1]:
                        new_start = token_starts[-1] + len(token) + 1
                        token_starts.append(new_start)

                    e1_span = get_span(e1_start, e1_end, token_starts)
                    e2_span = get_span(e2_start, e2_end, token_starts)

                    spans = {e1: e1_span, e2: e2_span}

                    sentences.append(Sentence(tokens,
                                              spans=spans,
                                              pmid=doc.pmid,
                                              ))
        return sentences

    def get_distance(self, pair, doc):
        token_starts = [0]
        for token in doc.tokens[:-1]:
            new_start = token_starts[-1] + len(token) + 1
            token_starts.append(new_start)


        e1_mentions = [a for a in doc.annotations if a.id == pair[0].id and a.type == pair[0].type]
        e2_mentions = [a for a in doc.annotations if a.id == pair[1].id and a.type == pair[1].type]

        min_distance = np.float('inf')

        for e1_mention in e1_mentions:
            for e2_mention in e2_mentions:
                dist = abs(e1_mention.start - e2_mention.start)

                min_distance = min(min_distance, dist)

        return min_distance
