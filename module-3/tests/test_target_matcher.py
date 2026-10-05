# tests/test_target_matcher.py
# ─────────────────────────────────────────────────────────────
# Module 3 – Milestone 3: Target Matching
# Unit tests for TargetMatcher and target_normalizer.
#
# No camera, no YOLO model, no GPU required — everything is pure
# Python with zero external dependencies beyond the project code.
#
# Test inventory
# ──────────────
# TestTargetNormalizer
#   1. normalize_target — strips article "the"
#   2. normalize_target — strips possessive "my"
#   3. normalize_target — alias "phone" -> "cell phone"
#   4. normalize_target — alias "sofa"  -> "couch"
#   5. normalize_target — unknown alias kept as-is
#   6. normalize_target — uppercase input is lowercased
#   7. normalize_target — empty string returns empty string
#   8. is_supported_class — valid COCO class returns True
#   9. is_supported_class — invalid class returns False
#
# TestTargetMatcherFound
#  10. target in detections -> found = True
#  11. multiple bottles -> highest-confidence one returned
#  12. multiple unrelated objects -> only target is selected
#
# TestTargetMatcherNotFound
#  13. target not in detections  -> found = False
#  14. empty detection list      -> found = False (no crash)
#
# TestTargetMatcherUnsupported
#  15. "watch" (not in COCO-80)  -> reason = "unsupported_class"
#  16. unsupported has found=False, conf=None, bbox=None
#
# TestTargetMatcherNormalization (end-to-end via TargetMatcher)
#  17. "my bottle" normalizes correctly and matches
#  18. Case-insensitive matching ("Bottle" detection matches "bottle" target)
#  19. "phone" alias resolves and matches "cell phone" detection
#
# TestTargetMatcherResultSchema
#  20. All required keys present in found result
#  21. All required keys present in not-found result
#  22. All required keys present in unsupported-class result
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import pytest

from matching.target_matcher import TargetMatcher
from matching.target_normalizer import is_supported_class, normalize_target


# ── Shared detection fixtures ──────────────────────────────────

def _det(class_name: str, confidence: float, bbox: list[int] | None = None) -> dict:
    """Build a minimal detection dict (mirrors YOLODetector output)."""
    return {
        "class_name": class_name,
        "class_id":   0,          # not used by matcher; value doesn't matter here
        "confidence": confidence,
        "bbox":       bbox or [10, 20, 100, 200],
    }


@pytest.fixture()
def matcher() -> TargetMatcher:
    """Fresh TargetMatcher instance for each test."""
    return TargetMatcher()


# ══════════════════════════════════════════════════════════════
# TestTargetNormalizer
# ══════════════════════════════════════════════════════════════

class TestTargetNormalizer:
    """Unit tests for the normalize_target / is_supported_class helpers."""

    def test_strips_article_the(self):
        """'the bottle' should become 'bottle'."""
        assert normalize_target("the bottle") == "bottle"

    def test_strips_possessive_my(self):
        """'my bottle' should become 'bottle'."""
        assert normalize_target("my bottle") == "bottle"

    def test_alias_phone_to_cell_phone(self):
        """'phone' is an alias for 'cell phone'."""
        assert normalize_target("phone") == "cell phone"

    def test_alias_sofa_to_couch(self):
        """'sofa' is an alias for 'couch'."""
        assert normalize_target("sofa") == "couch"

    def test_unknown_alias_returned_as_is(self):
        """An unknown word with no alias should be returned lowercase."""
        assert normalize_target("watch") == "watch"

    def test_uppercase_input_is_lowercased(self):
        """Input like 'BOTTLE' should normalize to 'bottle'."""
        assert normalize_target("BOTTLE") == "bottle"

    def test_empty_string_returns_empty(self):
        """An empty string should return an empty string."""
        assert normalize_target("") == ""

    def test_is_supported_class_bottle(self):
        """'bottle' is a valid COCO-80 class."""
        assert is_supported_class("bottle") is True

    def test_is_supported_class_watch_unsupported(self):
        """'watch' is NOT a COCO-80 class."""
        assert is_supported_class("watch") is False


# ══════════════════════════════════════════════════════════════
# TestTargetMatcherFound
# ══════════════════════════════════════════════════════════════

class TestTargetMatcherFound:
    """Tests where the target is present in detections."""

    def test_target_in_detections_found_true(self, matcher):
        """TEST 1: target = 'bottle', detections contain bottle -> found = True."""
        detections = [
            _det("person", 0.91),
            _det("bottle", 0.72),
        ]
        result = matcher.match("bottle", detections)
        assert result["found"] is True
        assert result["target"] == "bottle"

    def test_multiple_bottles_highest_confidence_selected(self, matcher):
        """TEST 3: multiple bottles -> highest-confidence bottle is returned."""
        detections = [
            _det("person", 0.91, [0, 0, 50, 50]),
            _det("bottle", 0.72, [100, 100, 200, 300]),
            _det("bottle", 0.86, [300, 100, 400, 300]),
        ]
        result = matcher.match("bottle", detections)
        assert result["found"] is True
        assert result["confidence"] == pytest.approx(0.86)
        assert result["bbox"] == [300, 100, 400, 300]

    def test_multiple_unrelated_objects_only_target_selected(self, matcher):
        """TEST 8: multiple unrelated objects -> only requested target is returned."""
        detections = [
            _det("person",     0.95, [0, 0, 50, 50]),
            _det("chair",      0.88, [50, 50, 150, 200]),
            _det("cell phone", 0.75, [200, 100, 350, 300]),
        ]
        result = matcher.match("cell phone", detections)
        assert result["found"] is True
        assert result["matched_class"] == "cell phone"
        assert result["confidence"] == pytest.approx(0.75)


# ══════════════════════════════════════════════════════════════
# TestTargetMatcherNotFound
# ══════════════════════════════════════════════════════════════

class TestTargetMatcherNotFound:
    """Tests where the target is absent from detections."""

    def test_target_not_in_detections_found_false(self, matcher):
        """TEST 2: target = 'bottle', detections do not contain bottle -> found = False."""
        detections = [
            _det("person", 0.91),
            _det("chair",  0.85),
        ]
        result = matcher.match("bottle", detections)
        assert result["found"] is False
        assert result["target"] == "bottle"
        assert result["confidence"] is None
        assert result["bbox"] is None

    def test_empty_detection_list_found_false(self, matcher):
        """TEST 7: empty detection list -> found = False without crashing."""
        result = matcher.match("bottle", [])
        assert result["found"] is False
        assert result["confidence"] is None
        assert result["bbox"] is None


# ══════════════════════════════════════════════════════════════
# TestTargetMatcherUnsupported
# ══════════════════════════════════════════════════════════════

class TestTargetMatcherUnsupported:
    """Tests for targets that are not in the COCO-80 class set."""

    def test_watch_returns_unsupported_class_reason(self, matcher):
        """TEST 6: 'watch' is not a COCO-80 class -> reason = 'unsupported_class'."""
        result = matcher.match("watch", [_det("clock", 0.91)])
        assert result["reason"] == "unsupported_class"

    def test_unsupported_class_has_correct_null_fields(self, matcher):
        """Unsupported target result must have found=False, conf=None, bbox=None."""
        result = matcher.match("watch", [])
        assert result["found"] is False
        assert result["confidence"] is None
        assert result["bbox"] is None
        assert result["matched_class"] is None
        assert result["reason"] == "unsupported_class"

    def test_unsupported_target_not_matched_to_similar_class(self, matcher):
        """'watch' must NOT be silently mapped to 'clock' or any other class."""
        detections = [_det("clock", 0.95)]
        result = matcher.match("watch", detections)
        # Should NOT be found (watch != clock)
        assert result["found"] is False
        assert result["reason"] == "unsupported_class"


# ══════════════════════════════════════════════════════════════
# TestTargetMatcherNormalization
# ══════════════════════════════════════════════════════════════

class TestTargetMatcherNormalization:
    """End-to-end normalization tests via TargetMatcher.match()."""

    def test_my_bottle_normalizes_and_matches(self, matcher):
        """TEST 4: 'my bottle' should normalize to 'bottle' and match."""
        detections = [_det("bottle", 0.80)]
        result = matcher.match("my bottle", detections)
        assert result["found"] is True
        assert result["target"] == "bottle"

    def test_case_insensitive_matching(self, matcher):
        """TEST 5: 'Bottle' (capitalized) target matches 'bottle' detection."""
        detections = [_det("bottle", 0.80)]
        result = matcher.match("Bottle", detections)
        assert result["found"] is True

    def test_phone_alias_matches_cell_phone_detection(self, matcher):
        """'phone' alias -> 'cell phone' -> matches 'cell phone' detection."""
        detections = [
            _det("person",     0.91),
            _det("cell phone", 0.78),
        ]
        result = matcher.match("phone", detections)
        assert result["found"] is True
        assert result["target"] == "cell phone"
        assert result["matched_class"] == "cell phone"

    def test_my_phone_alias_matches_cell_phone_detection(self, matcher):
        """'my phone' -> stripped 'phone' -> alias 'cell phone' -> matches."""
        detections = [_det("cell phone", 0.82)]
        result = matcher.match("my phone", detections)
        assert result["found"] is True
        assert result["target"] == "cell phone"


# ══════════════════════════════════════════════════════════════
# TestTargetMatcherResultSchema
# ══════════════════════════════════════════════════════════════

class TestTargetMatcherResultSchema:
    """All result dicts must contain exactly the expected set of keys."""

    _EXPECTED_KEYS = {"target", "found", "confidence", "bbox", "matched_class", "reason"}

    def test_found_result_has_all_keys(self, matcher):
        """TEST 20: Found result contains all required schema keys."""
        result = matcher.match("bottle", [_det("bottle", 0.85)])
        assert set(result.keys()) == self._EXPECTED_KEYS

    def test_not_found_result_has_all_keys(self, matcher):
        """TEST 21: Not-found result contains all required schema keys."""
        result = matcher.match("bottle", [_det("person", 0.91)])
        assert set(result.keys()) == self._EXPECTED_KEYS

    def test_unsupported_result_has_all_keys(self, matcher):
        """TEST 22: Unsupported-class result contains all required schema keys."""
        result = matcher.match("watch", [])
        assert set(result.keys()) == self._EXPECTED_KEYS

    def test_found_result_types(self, matcher):
        """Found result field types must be correct."""
        bbox = [10, 20, 100, 200]
        result = matcher.match("bottle", [_det("bottle", 0.85, bbox)])
        assert isinstance(result["target"],        str)
        assert isinstance(result["found"],         bool)
        assert isinstance(result["confidence"],    float)
        assert isinstance(result["bbox"],          list)
        assert isinstance(result["matched_class"], str)
        assert result["reason"] is None

    def test_not_found_result_types(self, matcher):
        """Not-found result field types must be correct."""
        result = matcher.match("bottle", [])
        assert isinstance(result["target"], str)
        assert result["found"]         is False
        assert result["confidence"]    is None
        assert result["bbox"]          is None
        assert result["matched_class"] is None
        assert result["reason"]        is None
