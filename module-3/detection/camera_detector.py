# detection/camera_detector.py
# ─────────────────────────────────────────────────────────────
# Module 3 – Camera & YOLO Object Detection
# Live camera capture + YOLO inference + Assistive Navigation loop.
#
# This module is the "orchestrator" for Milestones 2, 3, 4 & 5.
# It wires together:
#   OpenCV camera capture  →  YOLODetector  →  visualizer
#                                           →  TargetMatcher (M3)
#                                           →  NavigationController (M4/M5)
#
# Modes:
#   1. M2 mode: run_camera_loop(detector)
#   2. M3 mode: run_camera_loop(detector, target_matcher, "bottle")
#   3. M4/M5 mode: run_camera_loop(detector, nav_controller=nav_controller)
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
from detection.visualizer import (
    draw_detections,
    draw_navigation_overlay,
    draw_status_bar,
    draw_target_overlay,
)
from detection.yolo_detector import YOLODetector

logger = logging.getLogger(__name__)

# ── Window title shown in the OpenCV display ──────────────────
_WINDOW_TITLE = "Module 3 — Assistive YOLO Navigation  (press Q to quit)"


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
    nav_controller: Any | None = None,
    camera_index: int | None = None,
) -> None:
    """
    Start the live camera capture + YOLO detection + assistive guidance loop.

    Parameters
    ----------
    detector : YOLODetector
        An already-initialised detector instance.
    target_matcher : TargetMatcher | None
        TargetMatcher instance (M3 mode).
    raw_target : str | None
        The raw target string (e.g. "bottle").
    nav_controller : NavigationController | None
        NavigationController instance (M4/M5 assistive mode).
    camera_index : int | None
        Webcam device index.  ``None`` falls back to the ``CAMERA_INDEX``
        config value (default 0).  Pass ``args.camera`` to honour the
        ``--camera`` CLI flag.
    """
    # BUG FIX B3: use the caller-supplied index; fall back to config constant.
    cam_idx = camera_index if camera_index is not None else CAMERA_INDEX
    cap = _open_camera(cam_idx, CAMERA_BACKEND)

    # Give OpenCV a named window so it can be resized freely.
    cv2.namedWindow(_WINDOW_TITLE, cv2.WINDOW_NORMAL)

    frame_count = 0
    fps_timer = time.perf_counter()
    fps = 0.0

    # Mode detection
    m4_mode = nav_controller is not None
    m3_mode = (not m4_mode) and (target_matcher is not None and raw_target)

    last_status_print_t = 0.0
    last_printed_status = ""

    print("\n" + "─" * 55)
    print("  YOLO Assistive Navigation running.")
    print(f"  Model confidence threshold : {detector.confidence_threshold:.2f}")
    if m4_mode:
        print(f"  Mode                       : ASSISTIVE GUIDANCE (M4/M5)")
        print(f"  Target object              : {nav_controller.target_name!r}")
    elif m3_mode:
        print(f"  Mode                       : TARGET MATCHING (M3)")
        print(f"  Target object              : {raw_target!r}")
    else:
        print(f"  Mode                       : ALL DETECTIONS (M2)")
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
            h, w = frame.shape[:2]

            # ── YOLO inference ────────────────────────────────
            detections: list[dict[str, Any]] = detector.detect(frame)

            # ── FPS calculation (rolling 30-frame window) ─────
            if frame_count % 30 == 0:
                now = time.perf_counter()
                fps = 30.0 / max(now - fps_timer, 1e-6)
                fps_timer = now

            # ── Process Step based on mode ────────────────────
            match_result: dict[str, Any] | None = None
            nav_step: Any | None = None

            if m4_mode:
                nav_step = nav_controller.process_frame(detections, w, h)
            elif m3_mode:
                match_result = target_matcher.match(raw_target, detections)

            # ── Visualisation ─────────────────────────────────
            draw_detections(frame, detections, BBOX_THICKNESS, FONT_SCALE)
            draw_status_bar(frame, len(detections), fps)

            if nav_step is not None:
                draw_navigation_overlay(frame, nav_step)
            elif match_result is not None:
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

            # ── Periodic / Cooldown Terminal Output ──────────
            now = time.perf_counter()
            if nav_step is not None:
                current_instruction = nav_step.spoken_instruction
                elapsed = now - last_status_print_t
                if elapsed >= TARGET_STATUS_COOLDOWN_S or current_instruction != last_printed_status:
                    print(f"\n  [NAV] State: {nav_step.state.value:<10s} | Instruction: {current_instruction}")
                    last_status_print_t = now
                    last_printed_status = current_instruction
            elif match_result is not None:
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


def _print_target_status(match_result: dict[str, Any]) -> None:
    """Print a concise target-match status block to the terminal."""
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
