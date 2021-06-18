from pedl.utils import replace_consistently
import numpy as np


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

