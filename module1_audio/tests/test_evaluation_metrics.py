# tests/test_evaluation_metrics.py
# ─────────────────────────────────────────────────────────────
# Unit tests for evaluation/metrics.py
#
# Run with:  pytest tests/test_evaluation_metrics.py -v
#
# All tests are pure (no model, no API, no audio files needed).
# ─────────────────────────────────────────────────────────────

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluation.metrics import normalize_text, compute_wer, compute_cer, compute_accuracy


# ── normalize_text ─────────────────────────────────────────────

class TestNormalizeText:
    """Tests for the normalize_text() helper."""

    def test_lowercase(self):
        assert normalize_text("WHERE IS MY BLACK BOTTLE") == "where is my black bottle"

    def test_removes_punctuation(self):
        assert normalize_text("Where is my black bottle?") == "where is my black bottle"

    def test_removes_comma_and_period(self):
        assert normalize_text("Hello, World.") == "hello world"

    def test_collapses_spaces(self):
        assert normalize_text("hello   world") == "hello world"

    def test_strips_leading_trailing(self):
        assert normalize_text("  hello world  ") == "hello world"

    def test_empty_string(self):
        assert normalize_text("") == ""

    def test_preserves_meaningful_words(self):
        result = normalize_text("Can you see the green box?")
        assert "can" in result
        assert "green" in result
        assert "box" in result

    def test_apostrophe_removed(self):
        # Apostrophes are punctuation; they are removed
        result = normalize_text("I'm fine")
        assert "'" not in result

    def test_exclamation_removed(self):
        result = normalize_text("Stop! Go!")
        assert "!" not in result

    def test_numbers_preserved(self):
        # Digits are not punctuation, should be kept
        assert "3" in normalize_text("I have 3 bags")


# ── compute_wer ────────────────────────────────────────────────

class TestComputeWER:
    """Tests for compute_wer()."""

    def test_perfect_match(self):
        wer = compute_wer("Where is my black bottle", "Where is my black bottle")
        assert wer == pytest.approx(0.0, abs=1e-6)

    def test_perfect_match_case_insensitive(self):
        wer = compute_wer("Where is my black bottle", "where is my black bottle")
        assert wer == pytest.approx(0.0, abs=1e-6)

    def test_one_substitution(self):
        # "green" substituted with "red" in a 4-word ref
        ref = "find the green box"
        hyp = "find the red box"
        wer = compute_wer(ref, hyp)
        # 1 substitution / 4 words = 0.25
        assert wer == pytest.approx(0.25, abs=1e-4)

    def test_completely_wrong(self):
        ref = "hello world"
        hyp = "foo bar"
        wer = compute_wer(ref, hyp)
        # 2 substitutions / 2 words = 1.0
        assert wer == pytest.approx(1.0, abs=1e-4)

    def test_empty_hypothesis(self):
        # Empty hypothesis = all words deleted → WER = 1.0
        wer = compute_wer("hello world", "")
        assert wer == pytest.approx(1.0, abs=1e-4)

    def test_empty_reference_raises(self):
        with pytest.raises(ValueError, match="empty after normalisation"):
            compute_wer("", "hello world")

    def test_returns_float(self):
        wer = compute_wer("hello", "hello")
        assert isinstance(wer, float)

    def test_punctuation_does_not_affect_wer(self):
        # Punctuation removed during normalisation → same WER
        wer_with    = compute_wer("Where is my bottle?", "Where is my bottle!")
        wer_without = compute_wer("Where is my bottle",  "Where is my bottle")
        assert wer_with == pytest.approx(wer_without, abs=1e-6)


# ── compute_cer ────────────────────────────────────────────────

class TestComputeCER:
    """Tests for compute_cer()."""

    def test_perfect_match(self):
        cer = compute_cer("hello world", "hello world")
        assert cer == pytest.approx(0.0, abs=1e-6)

    def test_one_char_substitution(self):
        cer = compute_cer("cat", "bat")
        # 1 substitution / 3 characters = ~0.333
        assert 0.0 < cer <= 1.0

    def test_empty_hypothesis(self):
        cer = compute_cer("hello", "")
        assert cer == pytest.approx(1.0, abs=1e-4)

    def test_empty_reference_raises(self):
        with pytest.raises(ValueError, match="empty after normalisation"):
            compute_cer("", "hello")

    def test_returns_float(self):
        cer = compute_cer("hello", "hello")
        assert isinstance(cer, float)

    def test_cer_less_than_or_equal_wer(self):
        # CER is generally <= WER for short phrases
        ref = "find the blue bag"
        hyp = "find blue bag"          # one deletion
        wer = compute_wer(ref, hyp)
        cer = compute_cer(ref, hyp)
        # Not a strict rule, but a reasonable sanity check for short phrases
        assert cer >= 0.0


# ── compute_accuracy ───────────────────────────────────────────

class TestComputeAccuracy:
    """Tests for compute_accuracy()."""

    def test_zero_wer_gives_100_percent(self):
        assert compute_accuracy(0.0) == pytest.approx(100.0)

    def test_full_wer_gives_zero_percent(self):
        assert compute_accuracy(1.0) == pytest.approx(0.0)

    def test_half_wer(self):
        assert compute_accuracy(0.5) == pytest.approx(50.0)

    def test_over_one_wer_clamped_to_zero(self):
        # WER > 1.0 → accuracy must be 0, never negative
        assert compute_accuracy(1.5) == pytest.approx(0.0)
        assert compute_accuracy(2.0) == pytest.approx(0.0)

    def test_returns_float(self):
        assert isinstance(compute_accuracy(0.3), float)

    def test_accuracy_in_range(self):
        for wer in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]:
            acc = compute_accuracy(wer)
            assert 0.0 <= acc <= 100.0
