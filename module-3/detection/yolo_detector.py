# detection/yolo_detector.py
# ─────────────────────────────────────────────────────────────
# Module 3 – Camera & YOLO Object Detection
# Reusable YOLO detector component.
#
# Design intent
# ─────────────
# This class is the ONLY place that knows about Ultralytics / YOLO.
# All other modules (camera loop, future navigation, etc.) call:
#
#       detections = detector.detect(frame)
#
# and receive a plain list of dicts — no YOLO types leak out.
#
# Each detection dict has the shape:
#   {
#       "class_name":  str,          # e.g. "bottle"
#       "class_id":    int,          # COCO class index
#       "confidence":  float,        # 0.0 – 1.0
#       "bbox":        [x1, y1, x2, y2]  # pixel coords, ints
#   }
#
# Future modules (target matching, navigation, obstacle avoidance)
# will consume this exact structure — do NOT change it without
# updating downstream consumers.
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── Module-level YOLO name ────────────────────────────────────
# Importing YOLO here (at module scope rather than inside __init__)
# lets unittest.mock.patch('detection.yolo_detector.YOLO') work
# correctly in the test suite without needing to import ultralytics
# at test-collection time.
# The try/except preserves the ImportError path for environments
# where ultralytics is not installed.
try:
    from ultralytics import YOLO  # type: ignore[import-untyped]
except ImportError:
    YOLO = None  # type: ignore[assignment,misc]


class YOLODetector:
    """
    Thin, reusable wrapper around an Ultralytics YOLO model.

    Parameters
    ----------
    model_path : str
        Path to the YOLO weights file (e.g. ``"yolov8n.pt"``).
        Ultralytics will auto-download standard model files on first use.
    confidence_threshold : float
        Detections with confidence below this value are discarded.
    img_size : int
        Square image size passed to YOLO inference (pixels).
    device : str
        Inference device — ``"cpu"``, ``"cuda"``, or ``"mps"``.
    """

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float,
        img_size: int,
        device: str,
    ) -> None:
        self._confidence_threshold = confidence_threshold
        self._img_size = img_size
        self._device = device

        logger.info(
            "Loading YOLO model: %s  (device=%s, conf=%.2f, imgsz=%d)",
            model_path,
            device,
            confidence_threshold,
            img_size,
        )

        if YOLO is None:
            raise ImportError(
                "Ultralytics is not installed. "
                "Run:  pip install ultralytics"
            )

        self._model = YOLO(model_path)
        # Warm up the model so the first inference is not slow.
        # A tiny blank frame is enough to initialise internal state.
        dummy = np.zeros((self._img_size, self._img_size, 3), dtype=np.uint8)
        self._model.predict(
            dummy,
            imgsz=self._img_size,
            conf=self._confidence_threshold,
            device=self._device,
            verbose=False,
        )
        logger.info("YOLO model loaded and warmed up.")

    # ── Public API ────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> list[dict[str, Any]]:
        """
        Run YOLO inference on a single BGR frame (as returned by OpenCV).

        Parameters
        ----------
        frame : np.ndarray
            A BGR image array with shape (H, W, 3).

        Returns
        -------
        list[dict]
            A list of detection dicts, each with keys:
            ``class_name``, ``class_id``, ``confidence``, ``bbox``.
            Returns an empty list if nothing is detected.
        """
        results = self._model.predict(
            frame,
            imgsz=self._img_size,
            conf=self._confidence_threshold,
            device=self._device,
            verbose=False,   # suppress per-frame console spam from Ultralytics
        )

        detections: list[dict[str, Any]] = []

        # results is always a list of length 1 for a single frame
        if not results:
            return detections

        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            return detections

        for box in result.boxes:
            # xyxy gives [x1, y1, x2, y2] in pixel coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            class_id   = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            class_name = result.names[class_id]

            detections.append(
                {
                    "class_name": class_name,
                    "class_id":   class_id,
                    "confidence": round(confidence, 4),
                    "bbox":       [int(x1), int(y1), int(x2), int(y2)],
                }
            )

        return detections

    # ── Properties ────────────────────────────────────────────

    @property
    def confidence_threshold(self) -> float:
        """Currently active confidence threshold."""
        return self._confidence_threshold

    @property
    def class_names(self) -> dict[int, str]:
        """
        Mapping from class ID → class name for the loaded model.
        Useful for future modules that need to look up names by ID.
        """
        return self._model.names
