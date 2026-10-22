# detection/open_vocab_detector.py
# ─────────────────────────────────────────────────────────────
# Module 2 – Open-vocabulary object detection (roadmap R3)
#
# Finds objects outside the closed 15-class vocabulary (watch, keys,
# slippers, …) with the same output format as YOLODetector:
#
#       detections = detector.detect(frame)
#       → [{"class_name", "class_id", "confidence", "bbox": [x1,y1,x2,y2]}]
#
# Supported model families (all via the installed Ultralytics package):
#
#   kind        weights example          how a target is chosen
#   ──────────  ───────────────────────  ───────────────────────────────
#   "yoloe-pf"  yoloe-26s-seg-pf.pt      built-in vocabulary (no text
#                                        encoder); target must be in it
#   "yoloe"     yoloe-26s-seg.pt         text prompt → MobileCLIP encoder
#   "world"     yolov8s-worldv2.pt       text prompt → CLIP ViT-B/32
#
# Text encoders are downloaded on first use into WEIGHTS_DIR
# (module2_vision/weights/, gitignored) instead of Ultralytics' global
# weights folder.
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import contextlib
import logging
import os
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from config.config import WEIGHTS_DIR

logger = logging.getLogger(__name__)

KINDS = ("yoloe-pf", "yoloe", "world")


def infer_kind(model_path: str | Path) -> str:
    """Guess the model family from the weights file name."""
    name = Path(model_path).name.lower()
    if "world" in name:
        return "world"
    if name.startswith("yoloe"):
        return "yoloe-pf" if "-pf" in name else "yoloe"
    raise ValueError(f"Not an open-vocabulary model: {model_path!r}")


def _normalise(name: str) -> str:
    """Lowercase; '_' and '-' count as spaces ("flip flop" matches "flip-flop")."""
    return " ".join(name.lower().replace("_", " ").replace("-", " ").split())


@contextlib.contextmanager
def _download_into(weights_dir: Path) -> Iterator[None]:
    """
    Make Ultralytics store text-encoder files in ``weights_dir``.

    CLIP uses ``ultralytics.nn.text_model.WEIGHTS_DIR``; MobileCLIP
    downloads into the current directory.  Both are redirected only for
    the duration of the block (no change to the user's global settings).
    """
    import ultralytics.nn.text_model as text_model

    weights_dir.mkdir(parents=True, exist_ok=True)
    old_cwd, old_dir = os.getcwd(), text_model.WEIGHTS_DIR
    text_model.WEIGHTS_DIR = weights_dir
    os.chdir(weights_dir)
    try:
        yield
    finally:
        os.chdir(old_cwd)
        text_model.WEIGHTS_DIR = old_dir


class OpenVocabDetector:
    """
    Open-vocabulary detector with the YOLODetector interface.

    Parameters
    ----------
    model_path : str
        Weights file (see table above).  A bare file name is resolved
        inside ``weights_dir`` and downloaded there if missing.
    confidence_threshold, img_size, device :
        As for YOLODetector.
    kind : str, optional
        One of KINDS; inferred from the file name when omitted.
    """

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float,
        img_size: int,
        device: str,
        kind: str | None = None,
        weights_dir: Path = WEIGHTS_DIR,
    ) -> None:
        self.kind = kind or infer_kind(model_path)
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {self.kind!r}")
        self._confidence_threshold = confidence_threshold
        self._img_size = img_size
        self._device = device
        self._weights_dir = Path(weights_dir)
        self._target: str | None = None
        self._target_indices: list[int] | None = None  # prompt-free: indices into the built-in vocabulary
        self._prompts: list[str] = []

        path = Path(model_path)
        if not path.is_absolute() and path.parent == Path("."):
            path = self._weights_dir / path
        self.model_path = str(path)

        from ultralytics import YOLOE, YOLOWorld  # type: ignore[import-untyped]

        logger.info("Loading open-vocabulary model %s (kind=%s, device=%s)", path, self.kind, device)
        self._model = YOLOWorld(str(path)) if self.kind == "world" else YOLOE(str(path))

        # Built-in vocabulary of a prompt-free model (name → index).
        self._vocab: dict[str, int] = {}
        if self.kind == "yoloe-pf":
            names = self._model.names
            items = names.items() if isinstance(names, dict) else enumerate(names)
            self._vocab = {_normalise(n): int(i) for i, n in items}

    # ── Target selection ──────────────────────────────────────

    def supports(self, name: str) -> bool:
        """True if this detector can search for ``name``."""
        if self.kind == "yoloe-pf":
            return self._vocab_index(name) is not None
        return bool(name and name.strip())

    def set_target(
        self,
        class_name: str | None,
        confidence: float | None = None,
        prompts: list[str] | None = None,
    ) -> None:
        """
        Restrict detection to ``class_name``.

        ``prompts`` (optional) are the words actually given to the model,
        e.g. ``["watch", "wristwatch"]``; a hit for any of them is reported
        as ``class_name``.  Default: ``[class_name]``.
        Text-prompt models encode the words (the first call downloads the
        text encoder); prompt-free models look them up in their vocabulary
        and skip words they don't know.
        ``confidence`` (optional) replaces the confidence threshold.
        """
        if confidence is not None:
            self._confidence_threshold = confidence
        if class_name is None:
            self._target, self._target_indices, self._prompts = None, None, []
            return

        target = _normalise(class_name)
        words = list(dict.fromkeys(_normalise(p) for p in (prompts or [class_name]) if p.strip()))
        if self.kind == "yoloe-pf":
            indices = [i for i in (self._vocab_index(w) for w in words) if i is not None]
            if not indices:
                raise ValueError(f"none of {words} is in this model's built-in vocabulary")
            self._target_indices = sorted(set(indices))
        else:
            with _download_into(self._weights_dir):
                self._model.set_classes(words)
        self._target = target
        self._prompts = words
        logger.info("Open-vocabulary target %r, prompts %s (kind=%s)", target, words, self.kind)

    @property
    def prompts(self) -> list[str]:
        """Words currently given to the model for the target."""
        return list(self._prompts)

    def _vocab_index(self, name: str) -> int | None:
        """
        Exact vocabulary match, then the plain "-s" singular ("keys" → "key").
        No "-es" rule: it would turn "glasses" (spectacles) into "glass".
        """
        key = _normalise(name)
        for candidate in (key, key[:-1] if key.endswith("s") else None):
            if candidate and candidate in self._vocab:
                return self._vocab[candidate]
        return None

    # ── Inference ─────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> list[dict[str, Any]]:
        """Run inference on one BGR frame; returns YOLODetector-style dicts."""
        if self._target is None:
            raise RuntimeError("Call set_target() before detect()")

        kwargs: dict[str, Any] = {}
        if self._target_indices is not None:
            kwargs["classes"] = self._target_indices
        results = self._model.predict(
            frame,
            imgsz=self._img_size,
            conf=self._confidence_threshold,
            device=self._device,
            verbose=False,
            **kwargs,
        )
        detections: list[dict[str, Any]] = []
        if not results or results[0].boxes is None:
            return detections
        result = results[0]
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            class_id = int(box.cls[0].item())
            detections.append(
                {
                    # Report the requested name, so the matcher compares like with like.
                    "class_name": self._target,
                    "class_id": class_id,
                    "confidence": round(float(box.conf[0].item()), 4),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                }
            )
        return detections

    # ── Properties ────────────────────────────────────────────

    @property
    def confidence_threshold(self) -> float:
        return self._confidence_threshold

    @property
    def class_names(self) -> dict[int, str]:
        """Current searchable names: the target only (empty before set_target)."""
        return {0: self._target} if self._target else {}

    @property
    def vocabulary(self) -> list[str] | None:
        """Built-in vocabulary of a prompt-free model; None for text-prompt models (any word)."""
        return sorted(self._vocab) if self.kind == "yoloe-pf" else None
