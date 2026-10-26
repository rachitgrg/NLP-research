from __future__ import annotations

# evaluation/whisper_adapter.py
# ─────────────────────────────────────────────────────────────
# Module 1 – Hearing & Understanding
# WhisperEvalAdapter: the Whisper configuration of the 2026-09-16
# ASR benchmark (EXP-02): CPU, int8, beam 5, translate, language
# auto-detect.  Since 2026-10-23 it is a thin subclass of the app's
# WhisperTranscriber (speech/whisper_stt.py), so benchmark and app
# share one implementation.
#
# Requires faster-whisper:
#     pip install faster-whisper
# ─────────────────────────────────────────────────────────────

from speech.whisper_stt import WhisperTranscriber


class WhisperEvalAdapter(WhisperTranscriber):
    """
    Whisper with the EXP-02 benchmark settings.

    Usage::

        adapter = WhisperEvalAdapter(model_size="tiny")
        result  = adapter.transcribe("path/to/audio.wav")
        # result == {"text": "Where is my black bottle?", "language": "en"}
    """

    def __init__(self, model_size: str = "tiny", **kwargs) -> None:
        """
        Args:
            model_size: One of "tiny", "base", "small", "medium", "large-v3".
            kwargs:     Overrides for WhisperTranscriber (e.g. device="cuda").
        """
        settings = dict(device="cpu", compute_type="int8", language="auto", beam_size=5)
        settings.update(kwargs)
        super().__init__(model_size=model_size, **settings)
