# detection/detector_factory.py
# ─────────────────────────────────────────────────────────────
# Module 2 – Choose the detector setup for a requested target (R3)
#
#   target in the closed-set model's classes  → closed-set model only
#   otherwise, open-vocabulary enabled and the
#   open-vocabulary model can search for it   → HybridDetector:
#                                               closed-set (obstacles)
#                                               + open-vocabulary (target)
#   otherwise                                 → unsupported
#
# main.py and assistant.py both call prepare_target_detector(), so the
# two entry points behave identically.
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from config.config import (
    INFERENCE_IMG_SIZE,
    OFFLINE_MODE,
    OPEN_VOCAB_CONFIDENCE_THRESHOLD,
    OPEN_VOCAB_ENABLED,
    OPEN_VOCAB_MODEL_PATH,
    OPEN_VOCAB_PROMPTS,
    TARGET_CONFIDENCE_THRESHOLD,
)
from matching.target_normalizer import is_supported_class

logger = logging.getLogger(__name__)


class HybridDetector:
    """
    Runs the closed-set detector (all its classes, e.g. people and chairs
    for obstacle warnings) and an open-vocabulary detector restricted to
    the target, and returns one merged detection list in the usual format.
    """

    def __init__(self, closed_set: Any, open_vocab: Any) -> None:
        self.closed_set = closed_set
        self.open_vocab = open_vocab

    def detect(self, frame: np.ndarray) -> list[dict[str, Any]]:
        return self.closed_set.detect(frame) + self.open_vocab.detect(frame)

    @property
    def confidence_threshold(self) -> float:
        return self.closed_set.confidence_threshold

    @property
    def class_names(self) -> dict[int, str]:
        """Closed-set classes plus the open-vocabulary target (so matching accepts it)."""
        names = dict(self.closed_set.class_names)
        next_id = max(names, default=-1) + 1
        for i, name in enumerate(self.open_vocab.class_names.values()):
            if name not in names.values():
                names[next_id + i] = name
        return names


@dataclass
class TargetSetup:
    detector: Any           # YOLODetector or HybridDetector
    mode: str               # "closed-set" | "open-vocab" | "unsupported"
    message: str            # one line for the console


def _default_open_vocab_factory(device: str) -> Any:
    from detection.open_vocab_detector import OpenVocabDetector, missing_files

    # Offline (D-16): fail fast instead of letting Ultralytics try a download.
    missing = missing_files(OPEN_VOCAB_MODEL_PATH) if OFFLINE_MODE else []
    if missing:
        raise FileNotFoundError(
            "offline mode, model files missing: " + "; ".join(missing)
            + " (run tools/offline_check.py)"
        )
    return OpenVocabDetector(
        OPEN_VOCAB_MODEL_PATH, OPEN_VOCAB_CONFIDENCE_THRESHOLD, INFERENCE_IMG_SIZE, device
    )


def prepare_target_detector(
    closed_set: Any,
    target: str,
    device: str,
    open_vocab_enabled: bool = OPEN_VOCAB_ENABLED,
    open_vocab_factory: Callable[[str], Any] | None = None,
    prompt_map: dict[str, list[str]] | None = None,
) -> TargetSetup:
    """
    Configure detection for ``target`` (already normalized).

    The open-vocabulary model is only loaded when the closed-set model
    cannot detect the target, so common targets cost nothing extra.
    """
    if is_supported_class(target, closed_set.class_names):
        closed_set.set_target(target, TARGET_CONFIDENCE_THRESHOLD)
        return TargetSetup(closed_set, "closed-set", f"{target!r}: closed-set model")

    if not open_vocab_enabled:
        return TargetSetup(closed_set, "unsupported",
                           f"{target!r}: not in the closed-set classes (open vocabulary disabled)")

    try:
        open_vocab = (open_vocab_factory or _default_open_vocab_factory)(device)
    except Exception as exc:  # missing weights / text encoder, no network, …
        logger.error("Open-vocabulary model could not be loaded: %s", exc)
        return TargetSetup(closed_set, "unsupported",
                           f"{target!r}: open-vocabulary model unavailable ({exc})")

    if not open_vocab.supports(target):
        return TargetSetup(closed_set, "unsupported",
                           f"{target!r}: not in the open-vocabulary model's vocabulary")

    # Prompt map (D-14), e.g. keys → "bunch of keys"; words this model can't
    # search for (prompt-free vocabulary) are dropped, falling back to the target.
    mapped = [p for p in (prompt_map if prompt_map is not None else OPEN_VOCAB_PROMPTS).get(target, [])
              if open_vocab.supports(p)]
    prompts = mapped or [target]
    open_vocab.set_target(target, prompts=prompts)
    return TargetSetup(HybridDetector(closed_set, open_vocab), "open-vocab",
                       f"{target!r}: open-vocabulary model ({open_vocab.kind}, prompt: "
                       f"{' + '.join(repr(p) for p in prompts)}) + closed-set model for obstacles")
