# tests/test_query_parser.py
import pytest
from nlp.keyword_extractor import load_nlp_model
from nlp.query_parser import parse_query

@pytest.fixture(scope="module")
def nlp():
    return load_nlp_model("en_core_web_sm")

def test_find_object_simple(nlp):
    # 1. "Where is my bottle?"
    r = parse_query(nlp, "Where is my bottle?")
    assert r["intent"] == "find_object"
    assert r["object"] == "bottle"
    assert r["attributes"] == {}

def test_find_object_with_color(nlp):
    # 2. "Where is my black bottle?"
    r = parse_query(nlp, "Where is my black bottle?")
    assert r["intent"] == "find_object"
    assert r["object"] == "bottle"
    assert r["attributes"] == {"color": "black"}

def test_find_object_with_size_and_color(nlp):
    # 3. "Find the big red bag."
    r = parse_query(nlp, "Find the big red bag.")
    assert r["intent"] == "find_object"
    assert r["object"] == "bag"
    assert r["attributes"] == {"size": "big", "color": "red"}

def test_find_object_multiple_attributes(nlp):
    # 4. "Where is the small blue bottle on the left?"
    r = parse_query(nlp, "Where is the small blue bottle on the left?")
    assert r["intent"] == "find_object"
    assert r["object"] == "bottle"
    assert r["attributes"] == {"size": "small", "color": "blue", "position": "left"}

def test_identify_object(nlp):
    # 5. "What is this?"
    r = parse_query(nlp, "What is this?")
    assert r["intent"] == "identify_object"
    
def test_describe_object(nlp):
    # 6. "What color is this?"
    r = parse_query(nlp, "What color is this?")
    assert r["intent"] == "describe_object"

def test_general_question(nlp):
    # "Tell me something about this."
    r = parse_query(nlp, "Tell me something about this.")
    assert r["intent"] == "general_question"
