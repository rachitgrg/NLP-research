# tools/offline_check.py
# ─────────────────────────────────────────────────────────────
# Module 2 – Offline pre-flight check (roadmap R9, decisions.md D-16)
#
# Lists every model file the configured assistant needs and whether it
# is on disk, so a session can run without internet:
#   closed-set YOLO weights, open-vocabulary weights + text encoder,
#   speech recogniser (Whisper in the Hugging Face cache), spaCy model,
#   and (for information) the installed reply voices per language.
# Nothing is downloaded.  Exit code 1 if anything required is missing.
#
# Usage (from module2_vision/):
#   python tools/offline_check.py
#   ASR_MODEL_SIZE=medium python tools/offline_check.py
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parent.parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from config.config import (  # noqa: E402  (sets the offline flags first)
    OFFLINE_ENV_FLAGS,
    OFFLINE_MODE,
    OPEN_VOCAB_ENABLED,
    OPEN_VOCAB_MODEL_PATH,
    YOLO_MODEL_PATH,
    add_module1_to_path,
)

add_module1_to_path()


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    required: bool = True


def _size(path: Path) -> str:
    """Human-readable size of a file or directory."""
    if path.is_dir():
        total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    else:
        total = path.stat().st_size
    return f"{total / 1e6:,.0f} MB"


def check_vision() -> list[Check]:
    checks = []
    path = Path(YOLO_MODEL_PATH)
    checks.append(Check("Closed-set detector", path.is_file(),
                        f"{path} ({_size(path)})" if path.is_file() else f"missing: {path}"))
    if OPEN_VOCAB_ENABLED:
        from detection.open_vocab_detector import missing_files, required_files

        missing = missing_files(OPEN_VOCAB_MODEL_PATH)
        present = [next(Path(p) for p in group if Path(p).is_file())
                   for group in required_files(OPEN_VOCAB_MODEL_PATH)
                   if any(Path(p).is_file() for p in group)]
        detail = ", ".join(f"{p.name} ({_size(p)})" for p in present)
        if missing:
            detail = "missing: " + "; ".join(missing)
        checks.append(Check("Open-vocabulary detector + text encoder", not missing, detail))
    return checks


def check_speech() -> list[Check]:
    from speech.asr_config import ASR_BACKEND, ASR_LANGUAGE, ASR_MODEL_SIZE

    if ASR_BACKEND == "sarvam":
        return [Check("Speech recogniser", not OFFLINE_MODE,
                      "Sarvam cloud API: needs internet (not offline-capable)")]
    if ASR_BACKEND != "whisper":
        return [Check("Speech recogniser", False, f"unknown ASR_BACKEND {ASR_BACKEND!r}")]
    try:
        from faster_whisper.utils import download_model

        path = Path(download_model(ASR_MODEL_SIZE, local_files_only=True))
        return [Check(f"Speech recogniser (Whisper {ASR_MODEL_SIZE}, language {ASR_LANGUAGE})",
                      True, f"{path} ({_size(path)})")]
    except Exception as exc:  # not in the cache
        return [Check(f"Speech recogniser (Whisper {ASR_MODEL_SIZE})", False,
                      f"not in the Hugging Face cache ({type(exc).__name__})")]


def check_parser() -> list[Check]:
    try:
        import spacy.util

        ok = spacy.util.is_package("en_core_web_sm")
    except ImportError:
        ok = False
    return [Check("Request parser (spaCy en_core_web_sm)", ok, "installed" if ok else "missing")]


def check_voices() -> list[Check]:
    """Installed speech voices (informational): replies need a voice per language."""
    from audio_feedback.tts_speaker import sapi_voices, voice_languages

    voices = sapi_voices()
    names = ", ".join(name.split(" - ")[0] for name, _ in voices) or "none listed"
    return [Check(f"Reply voices (languages: {', '.join(sorted(voice_languages()))})",
                  True, names, required=False)]


def run_checks() -> list[Check]:
    return check_vision() + check_speech() + check_parser() + check_voices()


def main() -> int:
    flags = ", ".join(f"{k}={__import__('os').environ.get(k, '')}" for k in OFFLINE_ENV_FLAGS)
    print(f"OFFLINE_MODE={OFFLINE_MODE}  ({flags})\n")
    checks = run_checks()
    width = max(len(c.name) for c in checks)
    for c in checks:
        mark = "OK     " if c.ok else ("MISSING" if c.required else "absent ")
        print(f"  [{mark}] {c.name:<{width}}  {c.detail}")
    failed = [c for c in checks if c.required and not c.ok]
    print(f"\n{'Ready to run offline.' if not failed else f'{len(failed)} required item(s) missing.'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
