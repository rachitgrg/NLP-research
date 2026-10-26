# speech/asr_config.py
# ─────────────────────────────────────────────────────────────
# Module 1 – Speech recognition (ASR) settings.
#
# All values can be overridden with environment variables of the
# same name.  Offline mode (roadmap R9, decisions.md D-16) is the
# default: speech is recognised on this machine and models are
# loaded from the local cache only.
# ─────────────────────────────────────────────────────────────

import os

# ── Offline mode ──────────────────────────────────────────────
# Same switch and library flags as module2_vision/config/config.py, so
# module1_audio/main.py is offline on its own as well.  Must run before
# faster_whisper / huggingface_hub are imported.
OFFLINE_MODE: bool = os.environ.get("OFFLINE_MODE", "true").lower() in ("true", "1", "yes")
OFFLINE_ENV_FLAGS: dict[str, str] = {
    "YOLO_OFFLINE": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}


def apply_offline_mode(enabled: bool = OFFLINE_MODE) -> None:
    """Set OFFLINE_ENV_FLAGS (values already in the environment win)."""
    if enabled:
        for name, value in OFFLINE_ENV_FLAGS.items():
            os.environ.setdefault(name, value)


apply_offline_mode()

# ── Recogniser ────────────────────────────────────────────────
# "whisper" — faster-whisper on this machine (offline, default)
# "sarvam"  — Sarvam Saaras v3 cloud API (needs internet and
#             SARVAM_API_KEY; only with OFFLINE_MODE=false)
ASR_BACKEND: str = os.environ.get("ASR_BACKEND", "whisper").strip().lower()

# Whisper model: tiny, base, small, medium, large-v3 (or a local path).
# large-v3-turbo is NOT suitable: it was not trained to translate.
ASR_MODEL_SIZE: str = os.environ.get("ASR_MODEL_SIZE", "large-v3")

# The user's spoken language as a Whisper code ("hi", "en", "ta", …), or
# "auto" to detect it per request.  Setting it avoids language-detection
# errors on short (2–3 s) requests.
ASR_LANGUAGE: str = os.environ.get("ASR_LANGUAGE", "auto").strip().lower()

# "cuda" / "cpu"; empty = cuda if available.
ASR_DEVICE: str = os.environ.get("ASR_DEVICE", "").strip().lower()
# CTranslate2 compute type; empty = int8_float16 on cuda, int8 on cpu.
ASR_COMPUTE_TYPE: str = os.environ.get("ASR_COMPUTE_TYPE", "").strip().lower()
ASR_BEAM_SIZE: int = int(os.environ.get("ASR_BEAM_SIZE", "5"))
