# tests/test_keywords.py
# ─────────────────────────────────────────────────────────────
# Unit tests for nlp/keyword_extractor.py
# Run with:  pytest tests/test_keywords.py -v
# ─────────────────────────────────────────────────────────────

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from nlp.keyword_extractor import load_nlp_model, extract_keywords


@pytest.fixture(scope="module")
def nlp():
    """Load the English spaCy model once for the entire test session."""
    return load_nlp_model("en_core_web_sm")


class TestEnglishKeywords:

    def test_simple_noun(self, nlp):
        r = extract_keywords(nlp, "Where is my bottle?")
        assert "bottle" in r["keywords"]
        assert r["main_keyword"] == "bottle"

    def test_adjective_before_noun(self, nlp):
        r = extract_keywords(nlp, "Where is my black bottle?")
        assert "black" in r["keywords"]
        assert "bottle" in r["keywords"]
        assert r["main_keyword"] == "bottle"

    def test_adjective_and_noun_two_words(self, nlp):
        r = extract_keywords(nlp, "Find my red chair.")
        assert "red" in r["keywords"]
        assert "chair" in r["keywords"]
        assert r["main_keyword"] == "chair"

    def test_no_duplicates(self, nlp):
        r = extract_keywords(nlp, "The bottle is a bottle.")
        assert r["keywords"].count("bottle") == 1

    def test_stopwords_excluded(self, nlp):
        r = extract_keywords(nlp, "Where is my bottle?")
        for word in ["where", "is", "my"]:
            assert word not in r["keywords"]

    def test_multiple_nouns(self, nlp):
        r = extract_keywords(nlp, "I lost my keys and my wallet.")
        assert "keys" in r["keywords"] or "key" in r["keywords"]
        assert "wallet" in r["keywords"]

    def test_empty_string(self, nlp):
        r = extract_keywords(nlp, "")
        assert r["keywords"] == []
        assert r["main_keyword"] is None

    def test_whitespace_only(self, nlp):
        r = extract_keywords(nlp, "   ")
        assert r["keywords"] == []
        assert r["main_keyword"] is None

    def test_returns_correct_types(self, nlp):
        r = extract_keywords(nlp, "Find my phone.")
        assert isinstance(r["keywords"], list)
        assert isinstance(r["main_keyword"], (str, type(None)))

    def test_punctuation_stripped(self, nlp):
        r = extract_keywords(nlp, "Where... is my bottle?!")
        assert "bottle" in r["keywords"]

    def test_stopwords_only_gives_no_keywords(self, nlp):
        # "where is my" → all stopwords for English
        r = extract_keywords(nlp, "Where is my?")
        # Should return empty or minimal — no meaningful noun
        assert r["main_keyword"] is None or r["main_keyword"] not in ["where", "is", "my"]
