# tests/test_whisper_stt.py
# ─────────────────────────────────────────────────────────────
# Unit tests for speech/whisper_stt.py and the transcriber factory
# (offline speech recognition, roadmap R9).
# faster_whisper.WhisperModel is replaced by a fake: no model files,
# GPU or network are needed.  Run with:  pytest tests/test_whisper_stt.py -v
# ─────────────────────────────────────────────────────────────

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import faster_whisper  # noqa: E402

from evaluation.whisper_adapter import WhisperEvalAdapter  # noqa: E402
from speech.transcriber_factory import create_transcriber  # noqa: E402
from speech.whisper_stt import WhisperTranscriber, load_audio  # noqa: E402


class FakeWhisperModel:
    """Stands in for faster_whisper.WhisperModel."""

    instances = []

    def __init__(self, model_size, device, compute_type, local_files_only):
        self.args = dict(model_size=model_size, device=device, compute_type=compute_type,
                         local_files_only=local_files_only)
        self.text = " Where is my bottle? "
        self.detected = "hi"
        self.calls = []
        FakeWhisperModel.instances.append(self)

    def transcribe(self, audio, **kwargs):
        self.calls.append((audio, kwargs))
        segments = iter([SimpleNamespace(text=self.text)]) if self.text else iter([])
        return segments, SimpleNamespace(language=self.detected)


@pytest.fixture()
def fake_model(monkeypatch):
    FakeWhisperModel.instances = []
    monkeypatch.setattr(faster_whisper, "WhisperModel", FakeWhisperModel)
    return FakeWhisperModel


@pytest.fixture()
def wav(tmp_path):
    path = tmp_path / "speech.wav"
    sf.write(path, np.zeros(16000, dtype=np.float32), 16000)
    return str(path)


def test_translates_and_reports_detected_language(fake_model, wav, no_network):
    tr = WhisperTranscriber(model_size="large-v3", device="cpu", language="auto")
    result = tr.transcribe(wav)
    assert result == {"text": "Where is my bottle?", "language": "hi"}
    audio, kwargs = fake_model.instances[0].calls[0]
    assert kwargs["task"] == "translate"
    assert kwargs["language"] is None          # auto-detect
    assert isinstance(audio, np.ndarray) and audio.dtype == np.float32


def test_language_hint_is_passed_and_reported(fake_model, wav, no_network):
    tr = WhisperTranscriber(model_size="large-v3", device="cpu", language="ta")
    assert tr.transcribe(wav)["language"] == "ta"
    assert fake_model.instances[0].calls[0][1]["language"] == "ta"


def test_offline_loads_local_files_only(fake_model, no_network):
    WhisperTranscriber(model_size="tiny", device="cpu", local_files_only=True)
    args = fake_model.instances[0].args
    assert args["local_files_only"] is True
    assert args["compute_type"] == "int8"


def test_cuda_default_compute_type(fake_model, monkeypatch, no_network):
    import speech.whisper_stt as ws

    monkeypatch.setattr(ws, "register_cuda_libraries", lambda: True)
    WhisperTranscriber(model_size="tiny", device="cuda")
    assert fake_model.instances[0].args["compute_type"] == "int8_float16"


def test_missing_model_offline_gives_clear_error(monkeypatch, no_network):
    def not_cached(*args, **kwargs):
        raise OSError("LocalEntryNotFoundError")

    monkeypatch.setattr(faster_whisper, "WhisperModel", not_cached)
    with pytest.raises(RuntimeError, match="not available offline"):
        WhisperTranscriber(model_size="large-v3", device="cpu", local_files_only=True)


def test_empty_transcript_raises(fake_model, wav):
    tr = WhisperTranscriber(model_size="tiny", device="cpu")
    fake_model.instances[0].text = ""
    with pytest.raises(RuntimeError, match="empty transcript"):
        tr.transcribe(wav)


def test_missing_audio_file_raises(fake_model):
    tr = WhisperTranscriber(model_size="tiny", device="cpu")
    with pytest.raises(FileNotFoundError):
        tr.transcribe("does_not_exist.wav")


def test_load_audio_mono_and_resampled(tmp_path):
    path = tmp_path / "stereo_8k.wav"
    sf.write(path, np.ones((8000, 2), dtype=np.float32) * 0.5, 8000)   # 1 s stereo at 8 kHz
    audio = load_audio(str(path))
    assert audio.ndim == 1 and audio.dtype == np.float32
    assert len(audio) == 16000
    assert np.allclose(audio, 0.5, atol=1e-3)


def test_eval_adapter_keeps_benchmark_settings(fake_model):
    adapter = WhisperEvalAdapter(model_size="base")
    assert fake_model.instances[0].args["device"] == "cpu"
    assert fake_model.instances[0].args["compute_type"] == "int8"
    assert adapter.language is None and adapter.beam_size == 5


# ── Factory ────────────────────────────────────────────────────

def test_factory_builds_whisper(fake_model, no_network):
    tr = create_transcriber("whisper", offline=True, model_size="small", device="cpu")
    assert isinstance(tr, WhisperTranscriber)
    assert fake_model.instances[0].args["model_size"] == "small"


def test_factory_refuses_cloud_backend_offline(no_network):
    with pytest.raises(EnvironmentError, match="OFFLINE_MODE"):
        create_transcriber("sarvam", offline=True)


def test_factory_builds_sarvam_when_online(monkeypatch):
    import speech.sarvam_stt as sarvam

    monkeypatch.setattr(sarvam, "SarvamTranscriber", lambda: "sarvam-client")
    assert create_transcriber("sarvam", offline=False) == "sarvam-client"


def test_factory_rejects_unknown_backend():
    with pytest.raises(ValueError, match="ASR_BACKEND"):
        create_transcriber("gemini")


def test_hearing_pipeline_uses_injected_transcriber(no_network):
    from pipeline.hearing_pipeline import HearingPipeline

    class FakeTranscriber:
        def transcribe(self, path):
            return {"text": "Where is my phone?", "language": "hi"}

    result = HearingPipeline(transcriber=FakeTranscriber()).run("unused.wav")
    assert result["language"] == "hi"
    assert result["object"] == "phone"
