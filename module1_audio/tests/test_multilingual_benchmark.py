# tests/test_multilingual_benchmark.py
# ─────────────────────────────────────────────────────────────
# Unit tests for the EXP-08 tooling: evaluation/multilingual_benchmark.py
# (scoring) and evaluation/record_multilingual.py (manifest).
# No audio, models or network are used.
# ─────────────────────────────────────────────────────────────

import csv
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluation.multilingual_benchmark import (  # noqa: E402
    base_language,
    load_normalizer,
    object_correct,
    parse_backend,
    summarise,
)
from evaluation.record_multilingual import (  # noqa: E402
    MANIFEST_FIELDS,
    PROMPTS,
    append_manifest,
    clip_name,
    load_prompts,
)


@pytest.fixture(scope="module")
def normalize():
    return load_normalizer()


def test_parse_backend_specs():
    assert parse_backend("whisper:large-v3+hint") == {
        "name": "whisper:large-v3+hint", "backend": "whisper", "model_size": "large-v3", "hint": True}
    assert parse_backend("sarvam")["model_size"] is None
    with pytest.raises(ValueError):
        parse_backend("whisper")
    with pytest.raises(ValueError):
        parse_backend("gemini")


def test_base_language():
    assert base_language("hi-IN") == "hi"
    assert base_language("en") == "en"
    assert base_language(None) == ""


def test_object_correct_uses_app_normaliser_and_alternatives(normalize):
    assert object_correct("phone", "cell phone", normalize)          # alias → cell phone
    assert object_correct("my keys", "keys|key", normalize)
    assert object_correct("spectacles", "glasses", normalize)
    assert not object_correct("clock", "watch", normalize)           # ghadi → clock is a miss
    assert not object_correct("", "bottle", normalize)


def _row(backend, language, correct, detected, latency="0.1", error=""):
    return {"backend": backend, "language": language, "correct": str(correct),
            "detected_language": detected, "latency_s": latency, "error": error}


def test_summarise_per_language_and_all():
    rows = [
        _row("w", "hi", True, "hi"), _row("w", "hi", False, "en"),
        _row("w", "en", True, "en", latency="0.3"),
        _row("w", "en", False, "", error="RuntimeError: x"),
    ]
    summary = {(s["backend"], s["language"]): s for s in summarise(rows)}
    assert summary[("w", "hi")]["object_accuracy"] == 0.5
    assert summary[("w", "hi")]["language_id_accuracy"] == 0.5
    assert summary[("w", "en")]["errors"] == 1
    assert summary[("w", "en")]["median_latency_s"] == 0.3          # failed clips excluded
    assert summary[("w", "all")]["n"] == 4


def test_prompt_sheet_is_complete():
    prompts = load_prompts(PROMPTS)
    assert len(prompts) == 20
    assert len({p["prompt_id"] for p in prompts}) == 20
    assert all(p["object"] and p["english"] and p["hindi_suggestion"] for p in prompts)


def test_append_manifest_replaces_rerecorded_clip(tmp_path):
    manifest = tmp_path / "recordings.csv"
    name = clip_name("hi", "S01", "quiet", "p01")
    assert name == "hi_S01_quiet_p01.wav"
    row = dict.fromkeys(MANIFEST_FIELDS, "x") | {"audio_file": f"multilingual/audio/{name}"}
    append_manifest(row, manifest)
    append_manifest(row | {"english": "again"}, manifest)
    with open(manifest, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1 and rows[0]["english"] == "again"


class _FakeStream:
    def __init__(self, active):
        self.active = active


def test_record_audio_gives_up_when_microphone_hangs(tmp_path, monkeypatch):
    import numpy as np
    from evaluation import record_samples

    stopped = []
    monkeypatch.setattr(record_samples.time, "sleep", lambda s: None)
    monkeypatch.setattr(record_samples.sd, "rec", lambda *a, **k: np.ones((10, 1), "float32"))
    monkeypatch.setattr(record_samples.sd, "get_stream", lambda: _FakeStream(active=True))
    monkeypatch.setattr(record_samples.sd, "stop", lambda **k: stopped.append(k))
    clock = iter(range(0, 1000, 5))
    monkeypatch.setattr(record_samples.time, "monotonic", lambda: next(clock))

    out = tmp_path / "clip.wav"
    assert record_samples.record_audio(str(out), duration=1) is False
    assert stopped and not out.exists()


def test_record_audio_saves_finished_recording(tmp_path, monkeypatch):
    import numpy as np
    from evaluation import record_samples

    monkeypatch.setattr(record_samples.time, "sleep", lambda s: None)
    monkeypatch.setattr(record_samples.sd, "rec", lambda *a, **k: np.full((1600, 1), 0.1, "float32"))
    monkeypatch.setattr(record_samples.sd, "get_stream", lambda: _FakeStream(active=False))

    out = tmp_path / "clip.wav"
    assert record_samples.record_audio(str(out), duration=1) is True
    assert out.exists()
