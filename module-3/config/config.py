# config/config.py
# ─────────────────────────────────────────────────────────────
# Module 3 – Camera & YOLO Object Detection
# Central configuration file.
#
# All tuneable parameters live here.
# Do NOT scatter these values throughout the codebase.
# ─────────────────────────────────────────────────────────────

import os
from pathlib import Path

# ── Project root (module-3/ directory) ────────────────────────
MODULE_ROOT = Path(__file__).resolve().parent.parent

# ── YOLO Model ────────────────────────────────────────────────
# YOLOv8n ("nano") — smallest / fastest pretrained model.
# Suitable for real-time inference on a CPU laptop.
# Ultralytics will auto-download this on first run if not found.
#
# To swap in a custom-trained model later, change this path:
#   YOLO_MODEL_PATH = r"C:\path\to\your\custom_best.pt"
YOLO_MODEL_PATH: str = os.environ.get(
    "YOLO_MODEL_PATH",
    "yolov8n.pt",          # downloaded to the CWD / Ultralytics cache on first run
)

# ── Detection ─────────────────────────────────────────────────
# Minimum confidence score to keep a detection (0.0 – 1.0).
CONFIDENCE_THRESHOLD: float = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.40"))

# Image size fed to YOLO (pixels, square).
# Smaller = faster; 640 is the standard YOLOv8 input size.
INFERENCE_IMG_SIZE: int = int(os.environ.get("INFERENCE_IMG_SIZE", "640"))

# ── Inference device ──────────────────────────────────────────
# "cpu"  — always works; safe default for any laptop.
# "cuda" — requires a CUDA-capable GPU + torch-cuda installed.
# "mps"  — Apple Silicon GPU (macOS only).
INFERENCE_DEVICE: str = os.environ.get("INFERENCE_DEVICE", "cpu")

# ── Camera ────────────────────────────────────────────────────
# Index of the webcam to open (0 = default/built-in camera).
CAMERA_INDEX: int = int(os.environ.get("CAMERA_INDEX", "0"))

# OpenCV backend hint (0 = auto-select).
# On Windows, CAP_DSHOW (700) is often more reliable than the default.
CAMERA_BACKEND: int = int(os.environ.get("CAMERA_BACKEND", "0"))

# ── Display ───────────────────────────────────────────────────
# Key (ord value) to press to quit the live feed.
QUIT_KEY: int = ord("q")

# Bounding-box line thickness (pixels).
BBOX_THICKNESS: int = 2

# Font scale for on-screen labels.
FONT_SCALE: float = 0.55

# ── Logging ───────────────────────────────────────────────────
# Print a detection summary every N frames (0 = every frame).
# Set higher to avoid flooding the terminal.
LOG_EVERY_N_FRAMES: int = int(os.environ.get("LOG_EVERY_N_FRAMES", "30"))
