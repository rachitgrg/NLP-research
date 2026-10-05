# detection/camera_detector.py
# ─────────────────────────────────────────────────────────────
# Module 3 – Camera & YOLO Object Detection
# Live camera capture + YOLO inference loop.
#
# This module is the "orchestrator" for Milestone 2.
# It wires together:
#   OpenCV camera capture  →  YOLODetector  →  visualizer
#
# The public entry point is:
#
#   from detection.camera_detector import run_camera_loop
#   run_camera_loop(detector)
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
)
from detection.visualizer import draw_detections, draw_status_bar
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


def run_camera_loop(detector: YOLODetector) -> None:
    """
    Start the live camera capture + YOLO detection loop.

    The loop:
    1. Reads a frame from the webcam.
    2. Passes it to ``detector.detect(frame)``.
    3. Draws bounding boxes and labels via the visualizer.
    4. Displays the annotated frame in an OpenCV window.
    5. Periodically logs detection summaries to the terminal.
    6. Exits cleanly when the user presses Q (or closes the window).

    The camera is always released and the window destroyed on exit,
    even if an exception is raised mid-loop.

    Parameters
    ----------
    detector : YOLODetector
        An already-initialised detector instance.
    """
    cap = _open_camera(CAMERA_INDEX, CAMERA_BACKEND)

    # Give OpenCV a named window so it can be resized freely.
    cv2.namedWindow(_WINDOW_TITLE, cv2.WINDOW_NORMAL)

    frame_count = 0
    fps_timer   = time.perf_counter()
    fps         = 0.0

    print("\n" + "─" * 55)
    print("  YOLO Live Detection running.")
    print(f"  Model confidence threshold : {detector.confidence_threshold:.2f}")
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

            # ── Visualisation ────────────────────────────────
            draw_detections(frame, detections, BBOX_THICKNESS, FONT_SCALE)
            draw_status_bar(frame, len(detections), fps)

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
