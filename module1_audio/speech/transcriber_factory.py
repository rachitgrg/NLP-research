# speech/transcriber_factory.py
# ─────────────────────────────────────────────────────────────
# Module 1 – Choose the speech recogniser (roadmap R9)
#
#   "whisper" → WhisperTranscriber (offline, default)
#   "sarvam"  → SarvamTranscriber  (cloud; needs OFFLINE_MODE=false)
#
# Every backend returns {"text": English, "language": code}, so the
# parser and the vision bridge do not depend on the choice.  Backends
# are imported lazily: offline use needs neither sarvamai nor an API key.
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

from typing import Any

from speech.asr_config import ASR_BACKEND, OFFLINE_MODE

BACKENDS = ("whisper", "sarvam")


def create_transcriber(
    backend: str = ASR_BACKEND,
    offline: bool = OFFLINE_MODE,
    **kwargs: Any,
) -> Any:
    """
    Build the transcriber for ``backend``.

    ``kwargs`` go to the backend's constructor (e.g. ``model_size`` or
    ``language`` for Whisper).

    Raises:
        ValueError:       Unknown backend.
        EnvironmentError: Cloud backend requested in offline mode, or
                          SARVAM_API_KEY missing.
    """
    backend = backend.strip().lower()
    if backend == "whisper":
        from speech.whisper_stt import WhisperTranscriber

        return WhisperTranscriber(**kwargs)
    if backend == "sarvam":
        if offline:
            raise EnvironmentError(
                "ASR_BACKEND=sarvam uses the internet, but OFFLINE_MODE is on.\n"
                "Set OFFLINE_MODE=false to use Sarvam, or ASR_BACKEND=whisper."
            )
        from speech.sarvam_stt import SarvamTranscriber

        return SarvamTranscriber(**kwargs)
    raise ValueError(f"ASR_BACKEND must be one of {BACKENDS}, got {backend!r}")
