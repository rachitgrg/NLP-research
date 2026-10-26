# tests/test_offline_mode.py
# ─────────────────────────────────────────────────────────────
# Module 2 – offline mode (roadmap R9, decisions.md D-16):
# library flags, model-file checks, fail-fast open-vocabulary
# loading, the pre-flight tool and the cached voice pipeline.
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import config.config as cfg
import detection.detector_factory as factory
from detection.open_vocab_detector import missing_files, required_files


def test_offline_flags_set_without_overriding(monkeypatch):
    for name in cfg.OFFLINE_ENV_FLAGS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")          # an explicit user setting wins
    cfg.apply_offline_mode(True)
    assert os.environ["YOLO_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert os.environ["HF_HUB_OFFLINE"] == "0"


def test_offline_flags_not_set_when_disabled(monkeypatch):
    for name in cfg.OFFLINE_ENV_FLAGS:
        monkeypatch.delenv(name, raising=False)
    cfg.apply_offline_mode(False)
    assert not any(name in os.environ for name in cfg.OFFLINE_ENV_FLAGS)


def test_required_files_per_model_kind(tmp_path):
    assert len(required_files("yoloe-26s-seg-pf.pt", tmp_path)) == 1          # no text encoder
    yoloe = required_files("yoloe-26s-seg.pt", tmp_path)
    assert Path(yoloe[1][0]).name == "mobileclip2_b.ts"
    world = required_files("yolov8s-worldv2.pt", tmp_path)
    assert world[1] == (str(tmp_path / "clip/ViT-B-32.pt"),)


def test_missing_files_reports_only_absent(tmp_path):
    (tmp_path / "yoloe-26s-seg.pt").write_bytes(b"x")
    missing = missing_files("yoloe-26s-seg.pt", tmp_path)
    assert len(missing) == 1 and "mobileclip2_b.ts" in missing[0]
    (tmp_path / "mobileclip_blt.ts").write_bytes(b"x")                      # an alternative is enough
    assert missing_files("yoloe-26s-seg.pt", tmp_path) == []


class _Closed:
    class_names = {0: "person", 1: "bottle"}
    confidence_threshold = 0.4

    def set_target(self, name, conf):
        pass


def test_open_vocab_fails_fast_offline_when_files_missing(tmp_path, no_network):
    with patch.object(factory, "OFFLINE_MODE", True), \
         patch.object(factory, "OPEN_VOCAB_MODEL_PATH", str(tmp_path / "yoloe-26s-seg.pt")):
        setup = factory.prepare_target_detector(_Closed(), "watch", "cpu")
    assert setup.mode == "unsupported"
    assert "offline mode, model files missing" in setup.message


def test_offline_check_reports_missing_items(monkeypatch, capsys, no_network):
    import tools.offline_check as oc

    monkeypatch.setattr(oc, "run_checks", lambda: [oc.Check("A", True, "ok"), oc.Check("B", False, "missing")])
    assert oc.main() == 1
    assert "1 required item(s) missing" in capsys.readouterr().out
    monkeypatch.setattr(oc, "run_checks", lambda: [oc.Check("A", True, "ok")])
    assert oc.main() == 0


def test_offline_check_runs_without_network(no_network):
    import tools.offline_check as oc

    names = [c.name for c in oc.run_checks()]
    assert any("Closed-set" in n for n in names)
    assert any("Speech recogniser" in n for n in names)


def test_voice_pipeline_loaded_once():
    import pipeline.m3_pipeline as bridge

    created = []

    class FakePipeline:
        def __init__(self):
            created.append(self)

        def run(self, audio_path):
            return {"text": "where is my phone", "language": "hi", "object": "phone"}

    with patch.object(bridge, "MODULE1_AVAILABLE", True), \
         patch.object(bridge, "_HearingPipeline", FakePipeline), \
         patch.object(bridge, "_pipeline_instance", None):
        bridge.run_voice_pipeline("a.wav")
        bridge.run_voice_pipeline("b.wav")
    assert len(created) == 1
