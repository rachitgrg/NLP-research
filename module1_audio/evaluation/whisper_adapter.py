from __future__ import annotations

# evaluation/whisper_adapter.py
# ─────────────────────────────────────────────────────────────
# Module 1 – Hearing & Understanding
# WhisperEvalAdapter: wraps faster-whisper for the evaluation
# benchmark, matching the interface of SarvamTranscriber so
# both models can be compared in evaluate.py.
#
# Requires faster-whisper:
#     pip install faster-whisper
# ─────────────────────────────────────────────────────────────

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from faster_whisper import WhisperModel


class WhisperEvalAdapter:
    """
    Thin wrapper around faster-whisper for use in the evaluation benchmark.

    Loads the Whisper model once at construction time and exposes a
    ``transcribe()`` method with the same signature as ``SarvamTranscriber``,
    so both can be evaluated with identical calling code.

    Usage::

        adapter = WhisperEvalAdapter(model_size="tiny")
        result  = adapter.transcribe("path/to/audio.wav")
        # result == {"text": "Where is my black bottle?", "language": "en"}
    """

    def __init__(self, model_size: str = "tiny") -> None:
        """
        Load the Whisper model once.

        Args:
            model_size: One of "tiny", "base", "small", "medium", "large".
                        Default is "tiny" for the evaluation benchmark.

        Raises:
            ImportError: If faster-whisper is not installed.
        """
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise ImportError(
                "faster-whisper is required for the Whisper evaluation adapter.\n"
                "Install it with:\n"
                "    pip install faster-whisper\n"
                "Or install all evaluation dependencies:\n"
                "    pip install -r requirements.txt"
            ) from exc

        print(f"[Whisper] Loading '{model_size}' model on CPU (int8)…")
        self._model = WhisperModel(model_size, device="cpu", compute_type="int8")
        print("[Whisper] Model ready.")

    def transcribe(self, audio_path: str) -> dict:
        """
        Transcribe an audio file, translating any language to English.

        Uses ``task="translate"`` so the output language is always English,
        matching Sarvam AI's ``mode="translate"`` behaviour.

        Args:
            audio_path: Absolute or relative path to a WAV/MP3 audio file.

        Returns:
            dict: {
                "text"     (str): English transcript / translation.
                "language" (str): Always ``"en"`` (Whisper translates to English).
            }

        Raises:
            FileNotFoundError: If ``audio_path`` does not exist.
            RuntimeError:      If Whisper returns an empty transcript.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        segments, _info = self._model.transcribe(
            audio_path,
            beam_size=5,
            task="translate",
            language=None,          # auto-detect source language
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=300),
        )

        full_text = " ".join(seg.text.strip() for seg in segments).strip()

        if not full_text:
            raise RuntimeError(
                "Whisper returned an empty transcript. "
                "Ensure the audio contains clear speech."
            )

        return {
            "text": full_text,
            "language": "en",
        }
