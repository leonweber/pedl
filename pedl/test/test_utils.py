import multiprocessing as mp

import numpy as np
import bioc

from pedl.utils import replace_consistently, get_pmid, replace_consistently_dict
from pedl.data_getter import DataGetterAPI
from pedl.pubtator_elasticsearch import _process_pubtator_files
from pedl.utils import root


def test_replace_consistently_dict():
    # Test case 1: Replace a single span
    text = "The quick brown fox jumps over the lazy dog."
    span_to_replacement = {(4, 9): "slow"}
    expected_text = "The slow brown fox jumps over the lazy dog."
    expected_span_to_replacement = {(4, 9): (4, 8)}
    result_text, result_span_to_replacement = replace_consistently_dict(text, span_to_replacement)
    assert result_text == expected_text
    assert result_span_to_replacement == expected_span_to_replacement

    # Test case 2: Replace multiple non-overlapping spans
    text = "Hello world!"
    span_to_replacement = {(0, 5): "Hi", (6, 12): "there"}
    expected_text = "Hi there"
    expected_span_to_replacement = {(0, 5): (0, 2), (6, 12): (3, 8)}
    result_text, result_span_to_replacement = replace_consistently_dict(text, span_to_replacement)
    assert result_text == expected_text
    assert result_span_to_replacement == expected_span_to_replacement

    # Test case 3: Replace a span that doesn't exist in the text
    text = "The quick brown fox jumps over the lazy dog."
    span_to_replacement = {(100, 105): "slow"}
    try:
        result_text, result_span_to_replacement = replace_consistently_dict(text, span_to_replacement)
    except Exception as e:
        assert isinstance(e, AssertionError)

    # Test case 4: Replace a span with an empty string
    text = "Hello world!"
    span_to_replacement = {(0, 6): "", (6, 12): ""}
    expected_text = ""
    expected_span_to_replacement = {(0, 6): (0, 0), (6, 12): (0, 0)}
    result_text, result_span_to_replacement = replace_consistently_dict(text, span_to_replacement)
    assert result_text == expected_text
    assert result_span_to_replacement == expected_span_to_replacement

    # Test case 5: Regression
    text = "89-year-old female with biopsy-proven 1.3 cm right breast invasive ductal carcinoma, Nottingham histologic grade 2, ER/PR + HER2 -, presents for cryoablation."
    span_to_replacement = {(116, 119): '<gene1/>', (119, 121): '<gene2/>', (124, 128): '<gene3/>'}
    expected_text = "89-year-old female with biopsy-proven 1.3 cm right breast invasive ductal carcinoma, Nottingham histologic grade 2, <gene1/><gene2/> + <gene3/> -, presents for cryoablation."
    expected_span_to_replacement = {(116, 119): (116, 124), (119, 121): (124, 132), (124, 128): (135, 143)}
    result_text, result_span_to_replacement = replace_consistently_dict(text, span_to_replacement)
    assert result_text == expected_text
    assert result_span_to_replacement == expected_span_to_replacement



def test_replace_consistently():
    text = "Some roses might be reddish in nature"
    ent1 = "roses"
    ent2 = "red"
    ent3 = "nature"

    rep1 = "<e1>roses</e1>"
    rep2 = "<e2>red</e2>"
    rep3 = "<entity1/>"
    offsets = np.array([text.index(ent1), text.index(ent2), text.index(ent3)])
    lengths = [len(ent1), len(ent2), len(ent3)]

    assert text[offsets[0]:offsets[0] + lengths[0]] == ent1
    assert text[offsets[1]:offsets[1] + lengths[1]] == ent2
    assert text[offsets[2]:offsets[2] + lengths[2]] == ent3

    text, offsets = replace_consistently(offsets[0], lengths[0], rep1, text, offsets)
    assert text == "Some <e1>roses</e1> might be reddish in nature"
    offsets[0] += 4

    text, offsets = replace_consistently(offsets[1], lengths[1], rep2, text, offsets)
    assert text == "Some <e1>roses</e1> might be <e2>red</e2>dish in nature"
    offsets[1] += 4

    text, offsets = replace_consistently(offsets[2], lengths[2], "<entity1/>", text, offsets)
    assert text == "Some <e1>roses</e1> might be <e2>red</e2>dish in <entity1/>"

    text, offsets = replace_consistently(offsets[0], lengths[0], "<entity2/>", text, offsets)
    assert text == "Some <e1><entity2/></e1> might be <e2>red</e2>dish in <entity1/>"

    text, offsets = replace_consistently(offsets[1], lengths[1], "<entity2/>", text, offsets)
    assert text == "Some <e1><entity2/></e1> might be <e2><entity2/></e2>dish in <entity1/>"

