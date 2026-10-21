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


def draw_target_overlay(
    frame: np.ndarray,
    match_result: dict,
) -> np.ndarray:
    """
    Overlay a target-match status banner at the bottom of the frame.

    Milestone 3 addition — called only when a target is active.

    Colour coding:
    * **Green**  — target FOUND (high visibility, positive feedback).
    * **Red**    — target NOT FOUND.
    * **Orange** — target is an unsupported COCO class.

    Parameters
    ----------
    frame : np.ndarray
        BGR image to annotate (modified in-place and also returned).
    match_result : dict
        Result dict from ``TargetMatcher.match()``.  Expected keys:
        ``target``, ``found``, ``confidence``, ``bbox``,
        ``matched_class``, ``reason``.

    Returns
    -------
    np.ndarray
        The annotated frame (same array as *frame*).
    """
    h, w = frame.shape[:2]

    target = match_result.get("target", "")
    found  = match_result.get("found", False)
    reason = match_result.get("reason")
    conf   = match_result.get("confidence")
    bbox   = match_result.get("bbox")

    # ── Choose colour and label ────────────────────────────────
    if reason == "unsupported_class":
        colour     = (0, 165, 255)          # orange (BGR)
        status_str = f"TARGET: {target}  |  UNSUPPORTED CLASS"
    elif found:
        colour     = (0, 210, 60)           # green (BGR)
        conf_str   = f"{conf:.2f}" if conf is not None else "?"
        status_str = f"TARGET: {target}  |  FOUND  conf={conf_str}"
    else:
        colour     = (0, 0, 220)            # red (BGR)
        status_str = f"TARGET: {target}  |  NOT FOUND"

    # ── Banner background ──────────────────────────────────────
    banner_h = 30
    banner_y = h - banner_h
    cv2.rectangle(frame, (0, banner_y), (w, h), colour, cv2.FILLED)

    # ── Banner text ────────────────────────────────────────────
    cv2.putText(
        frame,
        status_str,
        (8, h - 9),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),    # white text on coloured banner
        1,
        cv2.LINE_AA,
    )

    # ── Highlight matched bbox with a thicker border ───────────
    if found and bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 4)

        # Cross-hair at bbox centre
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cross_len = 12
        cv2.line(frame, (cx - cross_len, cy), (cx + cross_len, cy), colour, 2)
        cv2.line(frame, (cx, cy - cross_len), (cx, cy + cross_len), colour, 2)

    return frame


def draw_navigation_overlay(
    frame: np.ndarray,
    nav_step: Any,
) -> np.ndarray:
    """
    Overlay navigation cues, zone guidelines, and obstacle warnings onto frame.

    Parameters
    ----------
    frame : np.ndarray
        BGR image to annotate.
    nav_step : NavigationStep
        Navigation step object from NavigationController.

    Returns
    -------
    np.ndarray
    """
    h, w = frame.shape[:2]

    # ── 1. Subtle Zone Guidelines ─────────────────────────────
    # Vertical sector marks at 20%, 40%, 60%, 80%
    zone_lines = [int(w * 0.20), int(w * 0.40), int(w * 0.60), int(w * 0.80)]
    for zx in zone_lines:
        cv2.line(frame, (zx, 0), (zx, h), (80, 80, 80), 1, cv2.LINE_AA)

    # ── 2. Obstacle Hazard Highlighting ───────────────────────
    if nav_step.obstacle_report.has_hazard:
        for hazard in nav_step.obstacle_report.hazards:
            hx1, hy1, hx2, hy2 = hazard.bbox
            # Bright warning red border
            cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (0, 0, 255), 3)
            warn_lbl = f"! HAZARD: {hazard.class_name.upper()} !"
            cv2.putText(
                frame,
                warn_lbl,
                (hx1, max(hy1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

    # ── 3. Target Highlighting & Crosshair ────────────────────
    loc = nav_step.spatial_location
    if nav_step.target_found and loc is not None:
        cx = int(loc.normalized_x * w)
        cy = int(loc.normalized_y * h)
        box_colour = (0, 255, 0) if not nav_step.is_arrived else (0, 255, 255)
        # Crosshair at target center
        cv2.circle(frame, (cx, cy), 8, box_colour, -1)
        cv2.line(frame, (cx - 20, cy), (cx + 20, cy), box_colour, 2)
        cv2.line(frame, (cx, cy - 20), (cx, cy + 20), box_colour, 2)

    # ── 4. Main Navigation Banner at Bottom ───────────────────
    banner_h = 42
    banner_y = h - banner_h

    # Color depends on state
    state_str = str(nav_step.state.value if hasattr(nav_step.state, "value") else nav_step.state)
    if state_str == "ARRIVED":
        bg_col = (0, 200, 255)       # Gold / Yellow
    elif state_str == "OBSTRUCTED":
        bg_col = (0, 0, 220)         # Red
    elif state_str == "GUIDING":
        bg_col = (0, 180, 50)        # Green
    elif state_str == "UNSUPPORTED":
        bg_col = (0, 140, 255)       # Orange
    else:
        bg_col = (60, 60, 60)        # Dark Slate for searching

    cv2.rectangle(frame, (0, banner_y), (w, h), bg_col, cv2.FILLED)

    instruction_text = nav_step.spoken_instruction
    label = f"[{state_str}]  {instruction_text}"
    cv2.putText(
        frame,
        label,
        (12, h - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return frame

