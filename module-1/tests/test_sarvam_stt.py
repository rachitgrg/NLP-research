# tests/test_sarvam_stt.py
# ─────────────────────────────────────────────────────────────
# Unit tests for speech/sarvam_stt.py
# Run with:  pytest tests/test_sarvam_stt.py -v
#
# All tests use unittest.mock — NO real Sarvam API calls are made.
# ─────────────────────────────────────────────────────────────

import os
import sys
import tempfile
import wave

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Helpers ────────────────────────────────────────────────────

def _create_silent_wav(path: str, duration_s: float = 1.0, sample_rate: int = 16000):
    """Write a short silent WAV file for transcription tests."""
    n_samples = int(duration_s * sample_rate)
    silence = np.zeros(n_samples, dtype=np.int16)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(silence.tobytes())


def _make_mock_response(transcript: str, language_code: str) -> MagicMock:
    """Build a mock Sarvam API response object."""
    mock_response = MagicMock()
    mock_response.transcript = transcript
    mock_response.language_code = language_code
    return mock_response


# ── Fixtures ───────────────────────────────────────────────────

@pytest.fixture()
def silent_wav(tmp_path):
    """Return a path to a temporary silent WAV file."""
    wav_path = str(tmp_path / "silent.wav")
    _create_silent_wav(wav_path)
    return wav_path


@pytest.fixture(autouse=True)
def set_fake_api_key(monkeypatch):
    """
    Inject a fake SARVAM_API_KEY for every test so the client
    constructor doesn't raise EnvironmentError.
    """
    monkeypatch.setenv("SARVAM_API_KEY", "test-fake-key-do-not-use")


# ── Tests ──────────────────────────────────────────────────────

class TestSarvamTranscriberEnglish:
    """Test 1 — English audio → English text returned as-is."""

    def test_english_response_text(self, silent_wav):
        mock_resp = _make_mock_response(
            transcript="Where is my black bottle?",
            language_code="en-IN",
        )
        with patch("speech.sarvam_stt.SarvamAI") as MockClient:
            MockClient.return_value.speech_to_text.transcribe.return_value = mock_resp
            from speech.sarvam_stt import SarvamTranscriber
            t = SarvamTranscriber()
            result = t.transcribe(silent_wav)

        assert result["text"] == "Where is my black bottle?"

    def test_english_response_language(self, silent_wav):
        mock_resp = _make_mock_response(
            transcript="Where is my black bottle?",
            language_code="en-IN",
        )
        with patch("speech.sarvam_stt.SarvamAI") as MockClient:
            MockClient.return_value.speech_to_text.transcribe.return_value = mock_resp
            from speech.sarvam_stt import SarvamTranscriber
            t = SarvamTranscriber()
            result = t.transcribe(silent_wav)

        assert result["language"] == "en-IN"

    def test_returns_dict(self, silent_wav):
        mock_resp = _make_mock_response("Hello.", "en-IN")
        with patch("speech.sarvam_stt.SarvamAI") as MockClient:
            MockClient.return_value.speech_to_text.transcribe.return_value = mock_resp
            from speech.sarvam_stt import SarvamTranscriber
            t = SarvamTranscriber()
            result = t.transcribe(silent_wav)

        assert isinstance(result, dict)
        assert "text" in result
        assert "language" in result


class TestSarvamTranscriberHindi:
    """Test 2 — Hindi audio → English translation returned."""

    def test_hindi_translated_to_english_text(self, silent_wav):
        mock_resp = _make_mock_response(
            transcript="Where is my black bottle?",
            language_code="hi-IN",
        )
        with patch("speech.sarvam_stt.SarvamAI") as MockClient:
            MockClient.return_value.speech_to_text.transcribe.return_value = mock_resp
            from speech.sarvam_stt import SarvamTranscriber
            t = SarvamTranscriber()
            result = t.transcribe(silent_wav)

        # text must be English (the translation), NOT the original Hindi
        assert result["text"] == "Where is my black bottle?"

    def test_hindi_detected_language(self, silent_wav):
        mock_resp = _make_mock_response(
            transcript="Where is my black bottle?",
            language_code="hi-IN",
        )
        with patch("speech.sarvam_stt.SarvamAI") as MockClient:
            MockClient.return_value.speech_to_text.transcribe.return_value = mock_resp
            from speech.sarvam_stt import SarvamTranscriber
            t = SarvamTranscriber()
            result = t.transcribe(silent_wav)

        assert result["language"] == "hi-IN"


class TestSarvamTranscriberMissingKey:
    """Test 3 — Missing SARVAM_API_KEY raises EnvironmentError."""

    def test_missing_key_raises(self, monkeypatch):
        # Remove the key from the process environment
        monkeypatch.delenv("SARVAM_API_KEY", raising=False)

        # Also patch load_dotenv so it cannot reload from the real .env file
        with patch("speech.sarvam_stt.SarvamAI"), \
             patch("dotenv.load_dotenv", return_value=False):
            import importlib
            import speech.sarvam_stt as stt_module
            importlib.reload(stt_module)
            with pytest.raises(EnvironmentError, match="SARVAM_API_KEY"):
                stt_module.SarvamTranscriber()


class TestSarvamTranscriberAPIFailure:
    """Test 4 — Sarvam API error is wrapped as a RuntimeError."""

    def test_api_error_raises_runtime_error(self, silent_wav):
        from sarvamai.core.api_error import ApiError

        with patch("speech.sarvam_stt.SarvamAI") as MockClient:
            MockClient.return_value.speech_to_text.transcribe.side_effect = ApiError(
                status_code=401,
                body="Unauthorized: invalid API key",
            )
            from speech.sarvam_stt import SarvamTranscriber
            t = SarvamTranscriber()
            with pytest.raises(RuntimeError, match="Sarvam AI API error"):
                t.transcribe(silent_wav)

    def test_network_error_raises_runtime_error(self, silent_wav):
        with patch("speech.sarvam_stt.SarvamAI") as MockClient:
            MockClient.return_value.speech_to_text.transcribe.side_effect = Exception(
                "Connection timeout"
            )
            from speech.sarvam_stt import SarvamTranscriber
            t = SarvamTranscriber()
            with pytest.raises(RuntimeError, match="Sarvam AI request failed"):
                t.transcribe(silent_wav)


class TestSarvamTranscriberEmptyTranscript:
    """Test 5 — Empty / None transcript raises RuntimeError."""

    def test_empty_string_raises(self, silent_wav):
        mock_resp = _make_mock_response(transcript="", language_code="en-IN")
        with patch("speech.sarvam_stt.SarvamAI") as MockClient:
            MockClient.return_value.speech_to_text.transcribe.return_value = mock_resp
            from speech.sarvam_stt import SarvamTranscriber
            t = SarvamTranscriber()
            with pytest.raises(RuntimeError, match="empty transcription"):
                t.transcribe(silent_wav)

    def test_whitespace_only_raises(self, silent_wav):
        mock_resp = _make_mock_response(transcript="   ", language_code="en-IN")
        with patch("speech.sarvam_stt.SarvamAI") as MockClient:
            MockClient.return_value.speech_to_text.transcribe.return_value = mock_resp
            from speech.sarvam_stt import SarvamTranscriber
            t = SarvamTranscriber()
            with pytest.raises(RuntimeError, match="empty transcription"):
                t.transcribe(silent_wav)

    def test_none_transcript_raises(self, silent_wav):
        mock_resp = _make_mock_response(transcript=None, language_code="en-IN")
        with patch("speech.sarvam_stt.SarvamAI") as MockClient:
            MockClient.return_value.speech_to_text.transcribe.return_value = mock_resp
            from speech.sarvam_stt import SarvamTranscriber
            t = SarvamTranscriber()
            with pytest.raises(RuntimeError, match="empty transcription"):
                t.transcribe(silent_wav)


class TestSarvamTranscriberFileNotFound:
    """File that doesn't exist should raise FileNotFoundError immediately."""

    def test_missing_file_raises(self):
        with patch("speech.sarvam_stt.SarvamAI"):
            from speech.sarvam_stt import SarvamTranscriber
            t = SarvamTranscriber()
            with pytest.raises(FileNotFoundError):
                t.transcribe("/nonexistent/path/audio.wav")
