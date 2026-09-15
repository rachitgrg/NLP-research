# tests/test_query_parser_extended.py
import pytest
from nlp.keyword_extractor import load_nlp_model
from nlp.query_parser import parse_query

@pytest.fixture(scope="module")
def nlp():
    return load_nlp_model("en_core_web_sm")

def test_simple_object_request(nlp):
    r = parse_query(nlp, "Where is my phone?")
    assert r["intent"] == "find_object"
    assert r["object"] == "phone"

def test_adjective_and_object(nlp):
    r = parse_query(nlp, "Where is my black bottle?")
    assert r["intent"] == "find_object"
    assert r["object"] == "bottle"
    assert r["attributes"].get("color") == "black"

def test_object_and_location(nlp):
    r = parse_query(nlp, "Where is my phone on the table?")
    assert r["intent"] == "find_object"
    assert r["object"] == "phone"
    assert "table" in r["keywords"]

def test_object_with_color(nlp):
    r = parse_query(nlp, "Where is the red bag?")
    assert r["intent"] == "find_object"
    assert r["object"] == "bag"
    assert r["attributes"].get("color") == "red"

def test_object_with_size(nlp):
    r = parse_query(nlp, "Find the big book.")
    assert r["intent"] == "find_object"
    assert r["object"] == "book"
    assert r["attributes"].get("size") == "big"

def test_object_with_position(nlp):
    r = parse_query(nlp, "Where is the laptop in front of me?")
    assert r["intent"] == "find_object"
    assert r["object"] == "laptop"
    assert r["attributes"].get("position") == "front"

def test_multiple_nouns(nlp):
    r = parse_query(nlp, "Can you find my bottle near the door?")
    assert r["intent"] == "find_object"
    assert r["object"] == "bottle"
    assert "door" in r["keywords"]

def test_conjunctions(nlp):
    r = parse_query(nlp, "Find the pen and the notebook.")
    assert r["intent"] == "find_object"
    # The parser currently picks one main object, likely 'notebook' or 'pen'
    # We just ensure it doesn't crash and picks a noun
    assert r["object"] in ["pen", "notebook"]

def test_multiple_attributes(nlp):
    r = parse_query(nlp, "Where is the small blue bottle on the left?")
    assert r["intent"] == "find_object"
    assert r["object"] == "bottle"
    assert r["attributes"].get("size") == "small"
    assert r["attributes"].get("color") == "blue"
    assert r["attributes"].get("position") == "left"
