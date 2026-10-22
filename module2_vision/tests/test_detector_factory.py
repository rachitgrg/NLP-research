# tests/test_detector_factory.py
# ─────────────────────────────────────────────────────────────
# Module 2 – detector selection for a target (R3) and HybridDetector
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import numpy as np

from detection.detector_factory import HybridDetector, prepare_target_detector
from matching.target_matcher import TargetMatcher
from matching.target_normalizer import normalize_target

FRAME = np.zeros((480, 640, 3), np.uint8)
CLOSED = {0: "person", 1: "chair", 5: "bottle"}


class FakeClosed:
    confidence_threshold = 0.40

    def __init__(self):
        self.class_names = dict(CLOSED)
        self.target = None

    def set_target(self, name, conf):
        self.target = (name, conf)

    def detect(self, frame):
        return [{"class_name": "person", "class_id": 0, "confidence": 0.9, "bbox": [0, 0, 100, 400]}]


class FakeOpenVocab:
    kind = "yoloe"

    def __init__(self, vocab=None):
        self.vocab = vocab           # None = any word
        self.target = None
        self.prompts = None

    def supports(self, name):
        return self.vocab is None or name in self.vocab

    def set_target(self, name, prompts=None):
        self.target = name
        self.prompts = prompts

    @property
    def class_names(self):
        return {0: self.target} if self.target else {}

    def detect(self, frame):
        return [{"class_name": self.target, "class_id": 0, "confidence": 0.6, "bbox": [300, 200, 340, 260]}]


def test_closed_set_target_uses_closed_model_only():
    closed, loaded = FakeClosed(), []
    setup = prepare_target_detector(closed, "bottle", "cpu", open_vocab_factory=lambda d: loaded.append(d))
    assert setup.mode == "closed-set" and setup.detector is closed
    assert closed.target == ("bottle", 0.25)
    assert loaded == []  # open-vocabulary model never loaded


def test_unknown_target_uses_hybrid_detector():
    ov = FakeOpenVocab()
    setup = prepare_target_detector(FakeClosed(), "watch", "cpu", open_vocab_factory=lambda d: ov)
    assert setup.mode == "open-vocab" and isinstance(setup.detector, HybridDetector)
    assert ov.target == "watch"
    names = [d["class_name"] for d in setup.detector.detect(FRAME)]
    assert names == ["person", "watch"]          # obstacles still detected
    assert "watch" in setup.detector.class_names.values()
    assert setup.detector.confidence_threshold == 0.40


def test_hybrid_target_is_accepted_by_matcher():
    setup = prepare_target_detector(FakeClosed(), "watch", "cpu", open_vocab_factory=lambda d: FakeOpenVocab())
    result = TargetMatcher(supported_classes=setup.detector.class_names).match("watch", setup.detector.detect(FRAME))
    assert result["found"] is True and result["bbox"] == [300, 200, 340, 260]


def test_disabled_open_vocab_reports_unsupported():
    setup = prepare_target_detector(FakeClosed(), "watch", "cpu", open_vocab_enabled=False)
    assert setup.mode == "unsupported"


def test_prompt_free_vocab_miss_reports_unsupported():
    setup = prepare_target_detector(FakeClosed(), "wallet", "cpu",
                                    open_vocab_factory=lambda d: FakeOpenVocab(vocab={"watch"}))
    assert setup.mode == "unsupported"


def test_open_vocab_load_failure_reports_unsupported():
    def broken(device):
        raise FileNotFoundError("mobileclip2_b.ts")

    setup = prepare_target_detector(FakeClosed(), "watch", "cpu", open_vocab_factory=broken)
    assert setup.mode == "unsupported" and "unavailable" in setup.message


def test_aliases_for_open_vocab_targets():
    assert normalize_target("my wristwatch") == "watch"
    assert normalize_target("spectacles") == "glasses"
    assert normalize_target("watch") == "watch"   # never mapped to "clock"


# ── Prompt map (D-14) ──────────────────────────────────────────

def test_keys_target_uses_mapped_prompt_but_keeps_its_name():
    ov = FakeOpenVocab()
    setup = prepare_target_detector(FakeClosed(), "keys", "cpu", open_vocab_factory=lambda d: ov,
                                    prompt_map={"keys": ["bunch of keys"]})
    assert ov.target == "keys" and ov.prompts == ["bunch of keys"]
    assert "bunch of keys" in setup.message
    assert setup.detector.detect(FRAME)[-1]["class_name"] == "keys"   # speech says "keys"


def test_unmapped_target_uses_itself_as_prompt():
    ov = FakeOpenVocab()
    prepare_target_detector(FakeClosed(), "watch", "cpu", open_vocab_factory=lambda d: ov,
                            prompt_map={"keys": ["bunch of keys"]})
    assert ov.prompts == ["watch"]


def test_mapped_prompt_unknown_to_model_falls_back_to_target():
    ov = FakeOpenVocab(vocab={"keys"})            # prompt-free style vocabulary
    setup = prepare_target_detector(FakeClosed(), "keys", "cpu", open_vocab_factory=lambda d: ov,
                                    prompt_map={"keys": ["bunch of keys"]})
    assert setup.mode == "open-vocab" and ov.prompts == ["keys"]


def test_default_prompt_map_and_env_format():
    from config.config import OPEN_VOCAB_PROMPTS, _parse_prompt_map
    assert OPEN_VOCAB_PROMPTS["keys"] == ["bunch of keys"] and OPEN_VOCAB_PROMPTS["key"] == ["bunch of keys"]
    assert _parse_prompt_map("keys=bunch of keys|keys; watch = watch ;bad") == {
        "keys": ["bunch of keys", "keys"], "watch": ["watch"]}
