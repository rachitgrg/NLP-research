# tests/test_phrases.py
# ─────────────────────────────────────────────────────────────
# Module 2 – spoken phrases per language (roadmap R9, phase 3):
# the phrase table, reply-language choice, and Hindi guidance
# through the locator, obstacle detector and navigation controller.
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

from guidance.obstacle_detector import ObstacleDetector
from guidance.phrases import PHRASES, choose_reply_language, object_name, phrase
from guidance.spatial_locator import SpatialLocator
from matching.target_matcher import TargetMatcher
from navigation.navigation_controller import NavigationController


def test_every_language_has_every_phrase():
    keys = set(PHRASES["en"])
    for lang, table in PHRASES.items():
        assert set(table) == keys, lang


def test_english_phrases_unchanged():
    assert phrase("target_left", "en", obj="bottle") == "Bottle, left."
    assert phrase("target_reached", "en", obj="cell phone") == \
        "Cell phone is directly in front of you. Stop and reach out."
    assert phrase("timeout", "en", obj="bottle") == "Could not find the bottle. Stopping the search."
    assert phrase("obstacle_ahead", "en", obs="dining table") == "Stop. Dining table ahead."
    assert phrase("cancelled") == "Search cancelled."


def test_hindi_phrases_use_hindi_object_names():
    assert phrase("target_left", "hi", obj="bottle") == "बोतल, बाईं ओर।"
    assert phrase("obstacle_ahead", "hi-IN", obs="chair") == "रुकिए। सामने कुर्सी है।"
    assert object_name("slippers", "hi") == "चप्पल"
    assert object_name("remote", "hi") == "remote"          # unknown word stays English


def test_unknown_language_falls_back_to_english():
    assert phrase("target_right", "xx", obj="cup") == "Cup, right."


def test_choose_reply_language():
    assert choose_reply_language("hi", {"en", "hi"}) == "hi"
    assert choose_reply_language("hi-IN", {"en", "hi"}) == "hi"
    assert choose_reply_language("hi", {"en"}) == "en"            # no Hindi voice installed
    assert choose_reply_language("ta", {"en", "ta"}) == "en"      # no Tamil phrase table yet
    assert choose_reply_language(None, {"en", "hi"}) == "en"
    assert choose_reply_language("en", {"en", "hi"}, setting="hi") == "hi"   # forced
    assert choose_reply_language("hi", {"en", "hi"}, setting="en") == "en"


def test_locator_formats_hindi_guidance():
    loc = SpatialLocator().locate([0, 100, 60, 200], 640, 480)       # far left, small box
    assert SpatialLocator.format_guidance(loc, "bottle", "hi") == "बोतल, काफ़ी बाईं ओर।"
    assert SpatialLocator.format_guidance(loc, "bottle") == "Bottle, far left."


def test_obstacle_warning_in_hindi():
    det = ObstacleDetector(language="hi")
    report = det.evaluate(
        [{"class_name": "chair", "class_id": 1, "confidence": 0.9, "bbox": [0, 100, 640, 480]}], 640, 480
    )
    assert report.primary_hazard.warning_text == "रुकिए। सामने कुर्सी है।"


class _Speaker:
    language = "hi"

    def __init__(self):
        self.said = []

    def speak(self, text, priority=None, force=False):
        self.said.append(text)
        return True


def test_navigation_controller_speaks_hindi():
    speaker = _Speaker()
    nav = NavigationController(
        target_name="bottle",
        matcher=TargetMatcher(supported_classes={0: "bottle"}),
        obstacle_detector=ObstacleDetector(enabled=False, language="hi"),
        speaker=speaker,
        language="hi",
    )
    step = nav.process_frame([], 640, 480)
    assert step.spoken_instruction == "बोतल नज़र में नहीं। धीरे से घूमिए।"
    step = nav.process_frame(
        [{"class_name": "bottle", "class_id": 0, "confidence": 0.9, "bbox": [560, 200, 620, 260]}], 640, 480
    )
    assert step.spoken_instruction == "बोतल, काफ़ी दाईं ओर।"
