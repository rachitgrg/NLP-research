# main.py
# ─────────────────────────────────────────────────────────────
# Module 3 – Camera & YOLO Object Detection
# Entry point — run the live YOLO camera pipeline.
#
# Usage:
#   python main.py
#   python main.py --model yolov8s.pt --conf 0.5
#   python main.py --camera 1 --device cuda
#
# All parameters default to the values in config/config.py.
# Command-line flags override config values for a single run only
# (they do NOT mutate config.py).
# ─────────────────────────────────────────────────────────────

import argparse
import logging
import sys

# ── Bootstrap logging before project imports ──────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Project imports ───────────────────────────────────────────
from config.config import (
    CAMERA_INDEX,
    CONFIDENCE_THRESHOLD,
    INFERENCE_DEVICE,
    INFERENCE_IMG_SIZE,
    YOLO_MODEL_PATH,
)
from detection.camera_detector import run_camera_loop
from detection.yolo_detector import YOLODetector


BANNER = """
╔══════════════════════════════════════════════════════╗
║      MODULE 3 — CAMERA & YOLO OBJECT DETECTION      ║
║                                                      ║
║  Real-time YOLOv8 detection on live webcam feed.     ║
║  Milestone 2: Camera + YOLO pipeline.                ║
╚══════════════════════════════════════════════════════╝
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Module 3: Live YOLO object detection via webcam."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=YOLO_MODEL_PATH,
        metavar="PATH",
        help=(
            f"Path to YOLO weights file. "
            f"Default: {YOLO_MODEL_PATH!r}. "
            "Ultralytics will download standard models (yolov8n.pt etc.) automatically."
        ),
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=CONFIDENCE_THRESHOLD,
        metavar="FLOAT",
        help=f"Confidence threshold (0–1). Default: {CONFIDENCE_THRESHOLD}.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=INFERENCE_IMG_SIZE,
        metavar="INT",
        help=f"YOLO inference image size (pixels). Default: {INFERENCE_IMG_SIZE}.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=INFERENCE_DEVICE,
        metavar="DEVICE",
        help=f"Inference device: 'cpu', 'cuda', 'mps'. Default: {INFERENCE_DEVICE!r}.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=CAMERA_INDEX,
        metavar="INDEX",
        help=f"Webcam device index. Default: {CAMERA_INDEX}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(BANNER)
    print(f"  Model      : {args.model}")
    print(f"  Confidence : {args.conf}")
    print(f"  Image size : {args.imgsz}")
    print(f"  Device     : {args.device}")
    print(f"  Camera idx : {args.camera}")
    print()

    # ── Initialise YOLO detector (loads model once) ────────────
    print("Loading YOLO model (first run will download weights ~6 MB)...")
    try:
        detector = YOLODetector(
            model_path=args.model,
            confidence_threshold=args.conf,
            img_size=args.imgsz,
            device=args.device,
        )
    except ImportError as exc:
        logger.error("%s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.error("Failed to load YOLO model: %s", exc)
        sys.exit(1)

    print("Model ready.\n")

    # ── Start live camera + detection loop ─────────────────────
    try:
        run_camera_loop(detector)
    except RuntimeError as exc:
        # Camera failed to open — print a friendly message
        logger.error("%s", exc)
        print(
            "\n[ERROR] Could not start camera. "
            "Make sure a webcam is connected and not used by another application.\n"
            "You can also try:\n"
            "  python main.py --camera 1\n"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
