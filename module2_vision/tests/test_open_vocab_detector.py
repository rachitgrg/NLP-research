# tests/test_open_vocab_detector.py
# ─────────────────────────────────────────────────────────────
# Module 2 – OpenVocabDetector (roadmap R3)
# Ultralytics model classes are replaced by fakes: no weights,
# text encoders or GPU are needed.
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import os
from unittest.mock import MagicMock

import numpy as np
import pytest

import detection.open_vocab_detector as ovd
from detection.open_vocab_detector import OpenVocabDetector, infer_kind


def _box(x1, y1, x2, y2, cls, conf):
    box = MagicMock()
    box.xyxy = [MagicMock()]
    box.xyxy[0].tolist.return_value = [x1, y1, x2, y2]
    box.cls = [MagicMock()]
    box.cls[0].item.return_value = cls
    box.conf = [MagicMock()]
    box.conf[0].item.return_value = conf
    return box


class FakeModel:
    """Stands in for ultralytics.YOLOE / YOLOWorld."""

    def __init__(self, path, names=None):
        self.path = path
        self.names = names if names is not None else {0: "object"}
        self.set_classes_calls = []
        self.predict_kwargs = None
        self.cwd_during_set_classes = None

    def set_classes(self, classes):
        self.set_classes_calls.append(list(classes))
        self.cwd_during_set_classes = os.getcwd()

    def predict(self, frame, **kwargs):
        self.predict_kwargs = kwargs
        result = MagicMock()
        result.boxes = [_box(10, 20, 110, 220, 7, 0.61)]
        return [result]


PF_VOCAB = {0: "person", 1: "key", 2: "watch", 3: "slipper", 4: "glass", 5: "sunglasses", 6: "flip-flop"}


@pytest.fixture()
def fake_ultralytics(monkeypatch):
    import ultralytics

    created = {}

    def yoloe(path):
        created["model"] = FakeModel(path, names=dict(PF_VOCAB) if "-pf" in path else None)
        return created["model"]

    def world(path):
        created["model"] = FakeModel(path)
        return created["model"]

    monkeypatch.setattr(ultralytics, "YOLOE", yoloe)
    monkeypatch.setattr(ultralytics, "YOLOWorld", world)
    return created


FRAME = np.zeros((480, 640, 3), np.uint8)


def test_infer_kind():
    assert infer_kind("yoloe-26s-seg-pf.pt") == "yoloe-pf"
    assert infer_kind("weights/yoloe-26s-seg.pt") == "yoloe"
    assert infer_kind("yolov8s-worldv2.pt") == "world"
    with pytest.raises(ValueError):
        infer_kind("best.pt")


class TestPromptFree:
    def _det(self, tmp_path):
        return OpenVocabDetector("yoloe-26s-seg-pf.pt", 0.25, 640, "cpu", weights_dir=tmp_path)

    def test_bare_name_resolves_into_weights_dir(self, fake_ultralytics, tmp_path):
        self._det(tmp_path)
        assert fake_ultralytics["model"].path == str(tmp_path / "yoloe-26s-seg-pf.pt")

    def test_vocabulary_lookup_and_plurals(self, fake_ultralytics, tmp_path):
        det = self._det(tmp_path)
        assert det.supports("watch") and det.supports("keys") and det.supports("Slippers")
        assert not det.supports("wallet")

    def test_hyphen_and_space_are_equivalent(self, fake_ultralytics, tmp_path):
        assert self._det(tmp_path).supports("flip flop")

    def test_glasses_does_not_map_to_drinking_glass(self, fake_ultralytics, tmp_path):
        assert not self._det(tmp_path).supports("glasses")

    def test_unknown_target_raises(self, fake_ultralytics, tmp_path):
        with pytest.raises(ValueError):
            self._det(tmp_path).set_target("wallet")

    def test_detect_restricts_to_target_class_and_reports_requested_name(self, fake_ultralytics, tmp_path):
        det = self._det(tmp_path)
        det.set_target("keys")
        dets = det.detect(FRAME)
        assert fake_ultralytics["model"].predict_kwargs["classes"] == [1]
        assert dets == [{"class_name": "keys", "class_id": 7, "confidence": 0.61, "bbox": [10, 20, 110, 220]}]
        assert det.class_names == {0: "keys"}


class TestTextPrompt:
    @pytest.mark.parametrize("weights", ["yoloe-26s-seg.pt", "yolov8s-worldv2.pt"])
    def test_any_word_supported_and_set_classes_called(self, fake_ultralytics, tmp_path, weights):
        det = OpenVocabDetector(weights, 0.25, 640, "cpu", weights_dir=tmp_path)
        assert det.supports("blue water bottle") and det.vocabulary is None
        cwd_before = os.getcwd()
        det.set_target("Wrist Watch")
        model = fake_ultralytics["model"]
        assert model.set_classes_calls == [["wrist watch"]]
        # Text encoder downloads happen inside weights_dir, then cwd is restored.
        assert os.path.samefile(model.cwd_during_set_classes, tmp_path)
        assert os.getcwd() == cwd_before
        assert "classes" not in (det.detect(FRAME) and model.predict_kwargs)

    def test_text_model_weights_dir_restored(self, fake_ultralytics, tmp_path):
        import ultralytics.nn.text_model as text_model
        before = text_model.WEIGHTS_DIR
        OpenVocabDetector("yoloe-26s-seg.pt", 0.25, 640, "cpu", weights_dir=tmp_path).set_target("watch")
        assert text_model.WEIGHTS_DIR == before


def test_detect_requires_target(fake_ultralytics, tmp_path):
    det = OpenVocabDetector("yoloe-26s-seg.pt", 0.25, 640, "cpu", weights_dir=tmp_path)
    with pytest.raises(RuntimeError):
        det.detect(FRAME)


class TestPromptSets:
    """Several prompt words for one target (prompt comparison, EXP-07)."""

    def test_text_model_gets_all_prompts_and_reports_target_name(self, fake_ultralytics, tmp_path):
        det = OpenVocabDetector("yoloe-26s-seg.pt", 0.25, 640, "cpu", weights_dir=tmp_path)
        det.set_target("watch", prompts=["watch", "Wrist-Watch", "watch"])
        assert fake_ultralytics["model"].set_classes_calls == [["watch", "wrist watch"]]
        assert det.prompts == ["watch", "wrist watch"]
        assert det.detect(FRAME)[0]["class_name"] == "watch"
        assert det.class_names == {0: "watch"}

    def test_prompt_free_uses_known_words_only(self, fake_ultralytics, tmp_path):
        det = OpenVocabDetector("yoloe-26s-seg-pf.pt", 0.25, 640, "cpu", weights_dir=tmp_path)
        det.set_target("keys", prompts=["keys", "keychain"])   # keychain not in the fake vocab
        det.detect(FRAME)
        assert fake_ultralytics["model"].predict_kwargs["classes"] == [1]

    def test_prompt_free_without_any_known_word_raises(self, fake_ultralytics, tmp_path):
        det = OpenVocabDetector("yoloe-26s-seg-pf.pt", 0.25, 640, "cpu", weights_dir=tmp_path)
        with pytest.raises(ValueError):
            det.set_target("wallet", prompts=["wallet", "purse"])
