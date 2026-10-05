# detection/visualizer.py
# ─────────────────────────────────────────────────────────────
# Module 3 – Camera & YOLO Object Detection
# Drawing utilities for annotating frames with detections.
#
# Kept separate from the detector so the rendering logic can be
# swapped / extended independently (e.g. different colour schemes,
# UI overlays, accessibility modes).
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

# Colour palette: one BGR colour per class ID (cycles automatically).
# These are visually distinct colours chosen for good contrast on
# both light and dark backgrounds.
_PALETTE: list[tuple[int, int, int]] = [
    (0,   200, 255),   # amber-yellow
    (0,   255, 128),   # spring-green
    (255, 100,   0),   # deep-blue
    (255,   0, 200),   # magenta
    (0,   128, 255),   # orange
    (128, 255,   0),   # chartreuse
    (200,   0, 255),   # violet
    (0,   255, 220),   # cyan-green
    (255, 220,   0),   # sky-blue
    (50,   50, 255),   # red
]


def _class_colour(class_id: int) -> tuple[int, int, int]:
    """Return a consistent BGR colour for a given class ID."""
    return _PALETTE[class_id % len(_PALETTE)]


def draw_detections(
    frame: np.ndarray,
    detections: list[dict[str, Any]],
    bbox_thickness: int = 2,
    font_scale: float = 0.55,
) -> np.ndarray:
    """
    Draw bounding boxes and labels for every detection onto *frame*.

    Parameters
    ----------
    frame : np.ndarray
        BGR image to annotate (modified in-place and also returned).
    detections : list[dict]
        Detection dicts produced by ``YOLODetector.detect()``.
    bbox_thickness : int
        Line thickness for bounding boxes.
    font_scale : float
        OpenCV font scale for label text.

    Returns
    -------
    np.ndarray
        The annotated frame (same array as *frame*).
    """
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        class_id   = det["class_id"]
        class_name = det["class_name"]
        conf       = det["confidence"]

        colour = _class_colour(class_id)
        label  = f"{class_name} {conf:.2f}"

        # ── Bounding box ───────────────────────────────────────
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, bbox_thickness)

        # ── Label background (filled rectangle) ───────────────
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )
        label_y_top = max(y1 - text_h - baseline - 4, 0)
        cv2.rectangle(
            frame,
            (x1, label_y_top),
            (x1 + text_w + 4, y1),
            colour,
            cv2.FILLED,
        )

        # ── Label text ────────────────────────────────────────
        # Dark text on coloured background for readability
        text_colour = (0, 0, 0)
        cv2.putText(
            frame,
            label,
            (x1 + 2, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_colour,
            1,
            cv2.LINE_AA,
        )

    return frame


def draw_status_bar(
    frame: np.ndarray,
    n_detections: int,
    fps: float,
) -> np.ndarray:
    """
    Overlay a small status bar at the top-left of the frame showing
    the detection count and current FPS.

    Parameters
    ----------
    frame : np.ndarray
        BGR image to annotate.
    n_detections : int
        Number of objects detected in the current frame.
    fps : float
        Measured frames-per-second.

    Returns
    -------
    np.ndarray
        The annotated frame (same array as *frame*).
    """
    status = f"Objects: {n_detections}  |  FPS: {fps:.1f}  |  [Q] Quit"
    cv2.putText(
        frame,
        status,
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return frame
