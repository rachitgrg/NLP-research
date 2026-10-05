# detection/camera_detector.py
# ─────────────────────────────────────────────────────────────
# Module 3 – Camera & YOLO Object Detection
# Live camera capture + YOLO inference loop.
#
# This module is the "orchestrator" for Milestone 2 and 3.
# It wires together:
#   OpenCV camera capture  →  YOLODetector  →  visualizer
#                                           →  TargetMatcher  (M3, optional)
#
# The public entry point is:
#
#   from detection.camera_detector import run_camera_loop
#   run_camera_loop(detector)                          # M2 mode
#   run_camera_loop(detector, target_matcher, "bottle") # M3 mode
#
# It deliberately does NOT contain any YOLO logic — that all lives
# inside YOLODetector so future modules can reuse detect() cleanly.
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import logging
import time
from typing import Any

import cv2

from config.config import (
    BBOX_THICKNESS,
    CAMERA_BACKEND,
    CAMERA_INDEX,
    FONT_SCALE,
    LOG_EVERY_N_FRAMES,
    QUIT_KEY,
    TARGET_STATUS_COOLDOWN_S,
)
from detection.visualizer import draw_detections, draw_status_bar, draw_target_overlay
from detection.yolo_detector import YOLODetector

logger = logging.getLogger(__name__)

# ── Window title shown in the OpenCV display ──────────────────
_WINDOW_TITLE = "Module 3 — YOLO Live Detection  (press Q to quit)"


def _open_camera(index: int, backend: int) -> cv2.VideoCapture:
    """
    Attempt to open the webcam and raise a RuntimeError on failure.

    Parameters
    ----------
    index : int
        Camera device index (0 = default camera).
    backend : int
        OpenCV backend flag (0 = auto; 700 = CAP_DSHOW on Windows).

    Returns
    -------
    cv2.VideoCapture
        An opened capture object.

    Raises
    ------
    RuntimeError
        If the camera could not be opened.
    """
    logger.info("Opening camera (index=%d, backend=%d)...", index, backend)

    if backend:
        cap = cv2.VideoCapture(index, backend)
    else:
        cap = cv2.VideoCapture(index)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera at index {index}. "
            "Check that a webcam is connected and not in use by another app."
        )

    logger.info("Camera opened successfully.")
    return cap


def run_camera_loop(
    detector: YOLODetector,
    target_matcher: Any | None = None,
    raw_target: str | None = None,
) -> None:
    """
    Start the live camera capture + YOLO detection loop.

    Milestone 2 (M2) mode — ``target_matcher`` is ``None``:
        The loop runs exactly as before: frame capture → YOLO
        inference → visualisation → Q to quit.

    Milestone 3 (M3) mode — ``target_matcher`` and ``raw_target`` supplied:
        In addition to the M2 pipeline, after each inference the
        loop calls ``target_matcher.match(raw_target, detections)``
        and:
        • Overlays the match status on the live video window.
        • Prints a terminal status message subject to a cooldown
          to avoid flooding.

    Parameters
    ----------
    detector : YOLODetector
        An already-initialised detector instance.
    target_matcher : TargetMatcher | None
        An already-initialised :class:`~matching.target_matcher.TargetMatcher`
        instance.  Pass ``None`` to run in pure M2 mode.
    raw_target : str | None
        The raw target string (e.g. ``"bottle"`` or ``"my phone"``).
        Required when ``target_matcher`` is not ``None``.
    """
    cap = _open_camera(CAMERA_INDEX, CAMERA_BACKEND)

    # Give OpenCV a named window so it can be resized freely.
    cv2.namedWindow(_WINDOW_TITLE, cv2.WINDOW_NORMAL)

    frame_count = 0
    fps_timer   = time.perf_counter()
    fps         = 0.0

    # ── M3: target-match state ────────────────────────────────
    # Track when we last printed a target-status message so we
    # don't flood the terminal at 20+ FPS.
    m3_mode              = target_matcher is not None and raw_target
    last_status_print_t  = 0.0       # epoch seconds
    last_printed_status  = ""        # "FOUND" / "NOT FOUND" / ""

    print("\n" + "─" * 55)
    print("  YOLO Live Detection running.")
    print(f"  Model confidence threshold : {detector.confidence_threshold:.2f}")
    if m3_mode:
        print(f"  Target object              : {raw_target!r}")
    print("  Press  Q  in the video window to quit.")
    print("─" * 55 + "\n")

    try:
        while True:
            # ── Capture frame ─────────────────────────────────
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.warning("Failed to read frame from camera. Retrying...")
                continue

            frame_count += 1

            # ── YOLO inference ────────────────────────────────
            detections: list[dict[str, Any]] = detector.detect(frame)

            # ── FPS calculation (rolling 30-frame window) ─────
            if frame_count % 30 == 0:
                now = time.perf_counter()
                fps = 30.0 / max(now - fps_timer, 1e-6)
                fps_timer = now

            # ── M3: target matching ───────────────────────────
            match_result: dict[str, Any] | None = None
            if m3_mode:
                match_result = target_matcher.match(raw_target, detections)

            # ── Visualisation ─────────────────────────────────
            draw_detections(frame, detections, BBOX_THICKNESS, FONT_SCALE)
            draw_status_bar(frame, len(detections), fps)
            if match_result is not None:
                draw_target_overlay(frame, match_result)

            cv2.imshow(_WINDOW_TITLE, frame)

            # ── Terminal logging (every N frames) ────────────
            if LOG_EVERY_N_FRAMES > 0 and frame_count % LOG_EVERY_N_FRAMES == 0:
                if detections:
                    print(f"[Frame {frame_count:>6d}]  {len(detections)} object(s) detected:")
                    for det in detections:
                        print(
                            f"   Detected: {det['class_name']:20s} | "
                            f"conf: {det['confidence']:.2f} | "
                            f"bbox: {det['bbox']}"
                        )
                else:
                    print(f"[Frame {frame_count:>6d}]  No objects detected.")

            # ── M3: print target status (with cooldown) ──────
            if match_result is not None:
                now = time.perf_counter()
                current_status = "FOUND" if match_result["found"] else "NOT FOUND"
                elapsed = now - last_status_print_t
                should_print = (
                    elapsed >= TARGET_STATUS_COOLDOWN_S
                    or current_status != last_printed_status
                )
                if should_print:
                    _print_target_status(match_result)
                    last_status_print_t = now
                    last_printed_status = current_status

            # ── Quit on Q key or window close ─────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == QUIT_KEY:
                print("\nQ pressed — stopping detection loop.")
                break

            # Also stop if the user closes the window via the X button
            if cv2.getWindowProperty(_WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                print("\nWindow closed — stopping detection loop.")
                break

    except KeyboardInterrupt:
        print("\n\nKeyboardInterrupt — stopping detection loop.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Camera released. Window destroyed.")
        print("\nCamera resources released. Goodbye!\n")


# ── Private helpers ────────────────────────────────────────────

def _print_target_status(match_result: dict[str, Any]) -> None:
    """
    Print a concise target-match status block to the terminal.

    Parameters
    ----------
    match_result : dict
        Result dict from ``TargetMatcher.match()``.
    """
    target = match_result["target"]
    reason = match_result.get("reason")

    if reason == "unsupported_class":
        print(
            f"\n  Target   : {target}\n"
            f"  Status   : UNSUPPORTED (not a COCO-80 class — YOLOv8n cannot detect this)\n"
        )
        return

    if match_result["found"]:
        conf = match_result["confidence"]
        bbox = match_result["bbox"]
        print(
            f"\n  Target   : {target}\n"
            f"  Status   : FOUND\n"
            f"  Confidence: {conf:.2f}\n"
            f"  BBox     : {bbox}\n"
        )
    else:
        print(
            f"\n  Target   : {target}\n"
            f"  Status   : NOT FOUND\n"
        )
