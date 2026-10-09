# pipeline/m3_pipeline.py
# ─────────────────────────────────────────────────────────────
# Module 3 – Milestone 3: Target Matching
# Integration bridge: Module 1 (voice/NLP) → Module 3 (target matching).
#
# This module is the ONLY place that knows about Module 1's
# HearingPipeline.  All other M3 components (TargetMatcher,
# camera loop, visualizer) remain completely independent.
#
# Design
# ──────
# The bridge exposes a single function:
#
#   extract_target_from_module1(result) -> dict
#
# that converts Module 1's output dict into the minimal structure
# that Module 3 needs:
#
#   {
#       "target":   str,         # the primary object noun (already English)
#       "language": str,         # BCP-47 source language code
#       "raw_text": str,         # original translated English text
#       "attributes": dict,      # color, size, position — preserved for M4+
#       "keywords":  list[str],  # all extracted keywords — preserved for M4+
#   }
#
# Integration flow
# ────────────────
#
#   Voice input
#       ↓
#   Module 1 – HearingPipeline.run(audio_path)
#       → {text, language, intent, object, attributes, keywords}
#       ↓
#   extract_target_from_module1(m1_result)
#       → {target, language, raw_text, attributes, keywords}
#       ↓
#   Module 3 – TargetMatcher.match(target, detections)
#       → {found, confidence, bbox, …}
#
# IMPORTANT
# ─────────
# Module 1 uses sys.path manipulation and has its own virtualenv.
# To avoid hard coupling, this module uses a try/except import so
# that M3 still works in isolation (e.g. via --target CLI flag)
# even when Module 1's dependencies (spaCy, sarvam-ai, etc.) are
# not installed in the M3 venv.
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Module 1 availability flag ────────────────────────────────
# Set to True only when the import succeeds below.
MODULE1_AVAILABLE: bool = False
_HearingPipeline = None  # placeholder — set after import


def _try_import_module1() -> None:
    """
    Attempt to import HearingPipeline from Module 1.

    Module 1 lives in a sibling directory (../module-1/).
    We add it to sys.path temporarily so that its internal imports
    (audio, speech, nlp, pipeline sub-packages) resolve correctly.

    This is intentionally non-fatal — M3 degrades gracefully when
    Module 1 is unavailable (e.g., running in a minimal venv or CI).
    """
    global MODULE1_AVAILABLE, _HearingPipeline

    module1_root = Path(__file__).resolve().parent.parent.parent / "module-1"

    if not module1_root.is_dir():
        logger.warning(
            "Module 1 not found at %s. Voice-input mode unavailable.", module1_root
        )
        return

    # Prepend module-1/ to sys.path if not already there
    module1_str = str(module1_root)
    if module1_str not in sys.path:
        sys.path.insert(0, module1_str)

    try:
        # Load .env from module-1/ so SARVAM_API_KEY is available
        from pathlib import Path as _P
        try:
            from dotenv import load_dotenv  # type: ignore
            _env = _P(module1_str) / ".env"
            load_dotenv(dotenv_path=_env, override=False)
        except ImportError:
            pass  # python-dotenv not installed — rely on shell env vars

        from pipeline.hearing_pipeline import HearingPipeline  # type: ignore
        _HearingPipeline = HearingPipeline
        MODULE1_AVAILABLE = True
        logger.info("Module 1 (HearingPipeline) imported successfully.")
    except ImportError as exc:
        logger.warning(
            "Could not import Module 1 (HearingPipeline): %s. "
            "Install Module 1 dependencies to enable voice-input mode.",
            exc,
        )
    except Exception as exc:
        logger.warning("Unexpected error importing Module 1: %s", exc)


# Run the import attempt once at module load time.
_try_import_module1()


# ── Public API ─────────────────────────────────────────────────

def extract_target_from_module1(m1_result: dict[str, Any]) -> dict[str, Any]:
    """
    Convert a Module 1 result dict into the target descriptor used by M3.

    This is a pure mapping function — it does not perform any inference.

    Module 1 returns::

        {
            "text":       str,        # English translation of spoken audio
            "language":   str,        # BCP-47 source language (e.g. "hi-IN")
            "intent":     str,        # e.g. "find_object"
            "object":     str | None, # primary noun extracted by spaCy
            "attributes": dict,       # {"color": "black", "size": "small", …}
            "keywords":   list[str],  # all extracted keywords
        }

    This function extracts the ``"object"`` field as the search target,
    preserving ``attributes`` and ``keywords`` for future milestones.

    Parameters
    ----------
    m1_result : dict
        The dict returned by ``HearingPipeline.run(audio_path)``.

    Returns
    -------
    dict
        ::

            {
                "target":     str,        # primary object noun (may be None → "")
                "language":   str,        # BCP-47 source language code
                "raw_text":   str,        # English text
                "attributes": dict,       # color/size/position — for M4+
                "keywords":   list[str],  # all keywords — for M4+
            }

    Notes
    -----
    * ``target`` is taken directly from ``m1_result["object"]``.
      Module 1's spaCy NLP has already stripped stop-words and
      possessives from the English translation, so the value is
      typically a clean noun (e.g. ``"bottle"``, not ``"my bottle"``).
    * ``TargetNormalizer.normalize_target()`` will be applied by
      ``TargetMatcher.match()`` — do NOT apply it here to avoid
      double-normalizing.
    * ``attributes`` contains things like ``{"color": "black"}`` —
      preserved for color-matching in a later milestone.
    """
    target = m1_result.get("object") or ""
    return {
        "target":     target,
        "language":   m1_result.get("language", "en"),
        "raw_text":   m1_result.get("text", ""),
        "attributes": m1_result.get("attributes", {}),
        "keywords":   m1_result.get("keywords", []),
    }


def run_voice_pipeline(audio_path: str) -> dict[str, Any] | None:
    """
    Run the full Module 1 voice pipeline on an audio file and return
    the M3 target descriptor.

    This is the top-level integration entry point called by ``main.py``
    when running in voice-input mode (``--voice`` flag).

    Parameters
    ----------
    audio_path : str
        Path to a WAV file recorded from the microphone.

    Returns
    -------
    dict | None
        Target descriptor dict (see :func:`extract_target_from_module1`)
        on success, or ``None`` if Module 1 is unavailable or failed.

    Raises
    ------
    RuntimeError
        Re-raised from Module 1 on transcription / API failure so that
        callers can decide how to handle it (retry, exit, etc.).
    """
    if not MODULE1_AVAILABLE or _HearingPipeline is None:
        raise RuntimeError(
            "Module 1 (HearingPipeline) is not available. "
            "Make sure Module 1 dependencies are installed:\n"
            "  pip install spacy sarvam-ai python-dotenv\n"
            "  python -m spacy download en_core_web_sm\n"
            "Or use --target <object> to skip voice input."
        )

    pipeline = _HearingPipeline()
    m1_result = pipeline.run(audio_path)
    return extract_target_from_module1(m1_result)
