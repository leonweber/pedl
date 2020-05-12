import itertools
import os
from bisect import bisect_right, bisect_left
from pathlib import Path
import argparse
import json
import scispacy
import spacy
from tqdm import tqdm
from mygene import MyGeneInfo

from .util import natural_language_to_uniprot
unmappable = set()

ENTITY_TYPES = {'Gene_or_gene_product', 'Protein'}
TYPE_MAPPING = {
    'Gene_expression': ['controls-expression-of'],
    'Translation': ['controls-expression-of'],
    'Transcription': ['controls-expression-of'],
    'Transport': ['controls-transport-of', 'controls-state-change-of'],
    'Phosphorylation': ['controls-phosphorylation-of', 'controls-state-change-of'],
    'Dephosphorylation': ['controls-phosphorylation-of', 'controls-state-change-of'],
    'Acetylation': ['controls-state-change-of'],
    'Deacetylation': ['controls-state-change-of'],
    'Ubiquitination': ['controls-state-change-of'],
    'Deubiquitination': ['controls-state-change-of'],
    'Hydroxylation': ['controls-state-change-of'],
    'Dehydroxylation': ['controls-state-change-of'],
    'Methylation': ['controls-state-change-of'],
    'Demethylation': ['controls-state-change-of'],
    'Glycosylation': ['controls-state-change-of'],
    'Deglycosylation': ['controls-state-change-of'],
    'Binding': ['in-complex-with'],
    'Dissociation': ['in-complex-with'],
    'Protein_modification': ['controls-state-change-of'],
    'Localization': ['controls-transport-of', 'controls-state-change-of'],
}


def get_span(start, end, token_starts):
    """
    Adapt annotations to token spans
    """

    # token_ends = [len(tokens[0])]
    # for token in tokens[1:]:
    # new_end = token_ends[-1] + len(token) + 1
    # token_ends.append(new_end)

    new_start = bisect_right(token_starts, start) - 1
    new_end = bisect_left(token_starts, end)

    return (new_start, new_end)


class Theme:
    registry = {}

    def __init__(self, id, start, end, mention, type):
        self.id = id
        self.start = start
        self.end = end
        self.mention = mention
        self.cause_of = []
        self.theme_of = []
        self.type = type

    def __str__(self):
        return f"{self.id}: {self.mention}"

    def __repr__(self):
        return str(self)

    @staticmethod
    def from_line(line):
        fields = line.strip().split('\t')
        id = fields[0]
        type, start, end = fields[1].split()
        start = int(start)
        end = int(end)
        mention = fields[2]

        return Theme(start=start, end=end, id=id, mention=mention, type=type)

    def register(self):
        self.registry[self.id] = self


def get_theme_or_event(id):
    if id.startswith('E'):
        return Event.registry[id]
    elif id.startswith('T'):
        return Theme.registry[id]
    else:
        raise ValueError(id)


class Event:
    registry = {}

    @classmethod
    def resolve_all_ids(cls):
        for event in cls.registry.values():
            event.resolve_ids()

    def __init__(self, id, themes, causes, products, mention, type):
        self.id = id
        self.themes = themes
        self.causes = causes
        self.products = products

        self.theme_of = []
        self.cause_of = []

        self.mention = mention
        self.type = type

    def __str__(self):
        return f"{self.id}:{self.type} Themes: {self.themes} Causes: {self.causes}"

    def __repr__(self):
        return str(self)

    @staticmethod
    def from_line(line):
        fields = line.strip().split('\t')
        id = fields[0]

        args = fields[1].split()
        type, mention = args[0].split(':')

        themes = set()
        causes = set()
        products = set()
        for arg in args[1:]:
            if arg.startswith('Theme'):
                themes.add(arg.split(':')[1])
            elif arg.startswith('Cause'):
                causes.add(arg.split(':')[1])
            elif arg.startswith('Participant'):
                continue
            elif arg.startswith('ToLoc'):
                continue
            elif arg.startswith('FromLoc'):
                continue
            elif arg.startswith('Product'):
                products.add(arg.split(':')[1])
            elif arg.startswith('Site'):
                continue
            elif arg.startswith('AtLoc'):
                continue
            elif arg.startswith('CSite'):
                continue
            elif arg.startswith('Sidechain'):
                continue
            elif arg.startswith('Contextgene'):
                continue
            else:
                raise ValueError(f"{arg}: {line}")

        return Event(id=id, themes=themes, causes=causes, mention=mention, type=type, products=products)

    def resolve_ids(self):
        resolved_themes = []
        for theme in self.themes:
            resolved_theme = get_theme_or_event(theme)
            resolved_themes.append(resolved_theme)
            resolved_theme.theme_of.append(self)
        self.themes = resolved_themes

        resolved_causes = []
        for cause in self.causes:
            resolved_cause = get_theme_or_event(cause)
            resolved_causes.append(resolved_cause)
            resolved_cause.cause_of.append(self)
        self.causes = resolved_causes

        resolved_products = []
        for product in self.products:
            resolved_product = get_theme_or_event(product)
            resolved_products.append(resolved_product)
        self.products = resolved_products

        self.mention = Theme.registry[self.mention]

    def register(self):
        self.registry[self.id] = self

    def get_regulators(self, processed_events):
        regulators = []
        for event in self.theme_of:
            if 'regulation' in event.type.lower() or 'catalysis' in event.type.lower():
                for cause in event.causes:
                    if isinstance(cause, Theme):
                        regulators.append(cause)
                    elif isinstance(cause, Event):
                        for theme in cause.themes:
                            if isinstance(theme, Theme):
                                regulators.append(theme)
                            elif isinstance(theme, Event):
                                if theme not in processed_events:
                                    processed_events.append(theme)
                                    regulators += theme.get_regulators(processed_events)

                        if cause not in processed_events:
                            processed_events.append(cause)
                            regulators += cause.get_regulators(processed_events)

                if event not in processed_events:
                    processed_events.append(event)
                    regulators += event.get_regulators(processed_events)

        return regulators


def parse(lines):
    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith('T'):
            Theme.from_line(line).register()
        elif line.startswith('E'):
            Event.from_line(line).register()


def get_possible_pairs(themes):
    possible_pairs = []

    for e1, e2 in itertools.combinations(themes, 2):
        if abs(e1.start - e2.start) < 300:
            possible_pairs.append((e1, e2))

    return possible_pairs


def get_mention(e1: Theme, e2: Theme, doc):
    if e1.start < e2.start:
        left_ent = e1
        left_ent_tag = 'e1'
        right_ent = e2
        right_ent_tag = 'e2'
    else:
        left_ent = e2
        left_ent_tag = 'e2'
        right_ent = e1
        right_ent_tag = 'e1'


    snippet_start = None
    snippet_end = None
    for sent in doc.sents:
        sent_start = sent[0].idx
        sent_end = sent[-1].idx + len(str(sent[-1]))
        if sent_end >= left_ent.start >= sent_start:
            snippet_start = sent_start

        if sent_end >= right_ent.end >= sent_start: # is sentence after right entity
            snippet_end = sent_end

    # adapt entity positions to text snippet
    left_start = left_ent.start - snippet_start
    left_end = left_ent.end - snippet_start
    right_start = right_ent.start - snippet_start
    right_end = right_ent.end - snippet_start

    text = str(doc)[snippet_start:snippet_end]

    # assert text[left_start:left_end] == left_ent.mention
    # assert text[right_start:right_end] == right_ent.mention

    new_text = text[:left_start] + f"<{left_ent_tag}>" + \
               text[left_start:left_end] + f"</{left_ent_tag}>" + \
               text[left_end:right_start] + f"<{right_ent_tag}>" + \
               text[right_start:right_end] + f"</{right_ent_tag}>" + \
               text[right_end:]

    return new_text


def transform(fname, transformed_data, mg):
    with open(fname + '.txt') as f:
        txt = f.read().strip()
    with open(fname + '.a1') as f:
        a1 = [l.strip() for l in f]
    with open(fname + '.a2') as f:
        a2 = [l.strip() for l in f]

    parse(a1 + a2)
    doc = nlp(txt)

    Event.resolve_all_ids()

    for e1, e2 in get_possible_pairs([t for t in Theme.registry.values() if t.type in ENTITY_TYPES]):
        transform_pair(e1, e2, relation_types=[], fname=fname, transformed_data=transformed_data,
                       doc=doc, is_direct=False, mg=mg)
        transform_pair(e2, e1, relation_types=[], fname=fname, transformed_data=transformed_data,
                       doc=doc, is_direct=False, mg=mg)

    event: Event
    for event_id, event in Event.registry.items():
        causes = []
        themes = [t for t in event.themes if isinstance(t, Theme)]

        if event.type not in TYPE_MAPPING:
            unmappable.add(event.type)
            continue
        else:
            relation_types = TYPE_MAPPING[event.type]

        if event.type == 'Binding':
            for e1, e2 in itertools.combinations(event.themes, 2):
                if e1.type not in ENTITY_TYPES or e2.type not in ENTITY_TYPES:
                    continue
                transform_pair(e1, e2, relation_types, fname, transformed_data, doc, is_direct=True)
                transform_pair(e2, e1, relation_types, fname, transformed_data, doc, is_direct=True)
        elif event.type == 'Dissociation':
            for e1, e2 in itertools.combinations(event.products, 2):
                if e1.type not in ENTITY_TYPES or e2.type not in ENTITY_TYPES:
                    continue
                transform_pair(e1, e2, relation_types, fname, transformed_data, doc, is_direct=True)
                transform_pair(e1, e1, relation_types, fname, transformed_data, doc, is_direct=True)
        else:
            causes += event.causes
            causes += event.get_regulators([])

            theme: Theme
            cause: Theme
            for theme in themes:
                for cause in causes:
                    if theme.type not in ENTITY_TYPES or cause.type not in ENTITY_TYPES:
                        continue
                    transform_pair(cause, theme, relation_types, fname, transformed_data, doc, is_direct=True)



    return transformed_data


def transform_pair(e1, e2, relation_types, fname, transformed_data, doc, mg, is_direct=False):
    e1_id = natural_language_to_uniprot(e1.mention, mg)
    e2_id = natural_language_to_uniprot(e2.mention, mg)

    pair = f"{e1_id},{e2_id}"
    if pair not in transformed_data:
        transformed_data[pair] = {
            'relations': set(),
            'mentions': set()
        }
    transformed_data[pair]['relations'].update(relation_types)
    mention = get_mention(e1=e1, e2=e2, doc=doc)
    if is_direct:
        transformed_data[pair]['mentions'] = set(m for m in transformed_data[pair]['mentions'] if m[0] != mention)
        transformed_data[pair]['mentions'].add(
            (mention, "direct", fname_to_pmid(fname))
    )
    else:
        transformed_data[pair]['mentions'].add(
            (mention, "distant", fname_to_pmid(fname))
        )


def fname_to_pmid(fname):
    x = itertools.dropwhile(lambda x: not str.isnumeric(x), str(os.path.basename(fname)))
    x = itertools.takewhile(lambda x: str.isnumeric(x), x)

    return ''.join(x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()

    data = Path(args.data)
    fnames = [str(fname.stem) for fname in data.glob('*.txt')]

    transformed_data = {}
    nlp = spacy.load('en_core_sci_sm', disable=['tagger'])
    mg = MyGeneInfo()

    for fname in tqdm(fnames):
        Event.registry = {}
        Theme.registry = {}
        transform(str(data / fname), transformed_data, mg=mg)

    json_compatible_data = {}
    for k, v in transformed_data.items():
        new_v = {'relations': list(v['relations']),
                 'mentions': [list(m) for m in v['mentions']]}
        json_compatible_data[k] = new_v

    os.makedirs(args.out.parent, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(json_compatible_data, f, indent=1)
    print("Did not transform because of missing mapping: ", unmappable)
