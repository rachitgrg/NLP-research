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

# ── Module and project roots ──────────────────────────────────
MODULE_ROOT = Path(__file__).resolve().parent.parent      # module2_vision/
PROJECT_ROOT = MODULE_ROOT.parent                          # NLP-Research/

# Module 1 (voice input + speech output) lives next to this module.
# audio_feedback (TTS) and the voice pipeline are imported from there.
MODULE1_ROOT = PROJECT_ROOT / "module1_audio"

# Downloaded model files (gitignored): YOLO weights and the text encoders
# used by open-vocabulary detectors.
WEIGHTS_DIR = MODULE_ROOT / "weights"


def add_module1_to_path() -> None:
    """Make module1_audio/ importable (for audio_feedback).

    Appended, not prepended: both modules have a ``pipeline`` package and
    this module's own packages must keep priority.
    """
    import sys

    if str(MODULE1_ROOT) not in sys.path:
        sys.path.append(str(MODULE1_ROOT))


# ── Expected Python environment ───────────────────────────────
# The integrated assistant needs the repository-root .venv (voice + vision
# packages).  Other interpreters on this machine lack some of them, which
# shows up later as confusing "No module named ..." errors.
EXPECTED_VENV = PROJECT_ROOT / ".venv"


def environment_warning() -> str | None:
    """Return a warning if not running from EXPECTED_VENV, else None."""
    import sys

    if Path(sys.prefix).resolve() == EXPECTED_VENV.resolve():
        return None
    return (
        "WARNING: not running from the project environment.\n"
        f"  Current Python : {sys.executable}\n"
        f"  Expected       : {EXPECTED_VENV / 'Scripts' / 'python.exe'}\n"
        "  Voice input and speech output may be unavailable.\n"
        "  Fix: run  ..\\.venv\\Scripts\\Activate.ps1  (from module2_vision), or call\n"
        "       ..\\.venv\\Scripts\\python.exe main.py ...  directly."
    )


# ── YOLO Model ────────────────────────────────────────────────
# Fine-tuned YOLOv8n on 15-class indoor assistive-navigation subset.
# Trained for 30 epochs on COCO train2014 → mAP@50 = 48.1%
#
# To revert to the generic pretrained model:
#   YOLO_MODEL_PATH = "weights/yolov8n.pt"   (run from module2_vision/)
YOLO_MODEL_PATH: str = os.environ.get(
    "YOLO_MODEL_PATH",
    str(MODULE_ROOT / "training" / "runs" / "train" / "weights" / "best.pt"),
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
# Default: "cuda" if this Python's torch build can see a GPU, otherwise
# "cpu".  (A CPU-only torch wheel cannot use the RTX 3050 even though the
# GPU exists, and requesting "cuda" there makes YOLO loading fail.)
def _default_device() -> str:
    try:
        import torch  # type: ignore
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


INFERENCE_DEVICE: str = os.environ.get("INFERENCE_DEVICE") or _default_device()

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

# ── Milestone 3 — Target Matching ────────────────────────────
# Minimum seconds between repeated terminal messages for the same
# target-match status (FOUND / NOT FOUND).  Prevents flooding the
# terminal when the camera loop runs at 20+ FPS.
# Set to 0 to print on every LOG_EVERY_N_FRAMES tick.
TARGET_STATUS_COOLDOWN_S: float = float(
    os.environ.get("TARGET_STATUS_COOLDOWN_S", "2.0")
)

# ── Milestone 4 — Spatial Guidance & Proximity ────────────────
# Normalized horizontal boundary thresholds (0.0 = left edge, 1.0 = right edge)
ZONE_FAR_LEFT_THRESHOLD: float = float(os.environ.get("ZONE_FAR_LEFT_THRESHOLD", "0.20"))
ZONE_LEFT_THRESHOLD: float = float(os.environ.get("ZONE_LEFT_THRESHOLD", "0.40"))
ZONE_RIGHT_THRESHOLD: float = float(os.environ.get("ZONE_RIGHT_THRESHOLD", "0.60"))
ZONE_FAR_RIGHT_THRESHOLD: float = float(os.environ.get("ZONE_FAR_RIGHT_THRESHOLD", "0.80"))

# Proximity area ratio thresholds (bbox_area / frame_area)
PROXIMITY_FAR_THRESHOLD: float = float(os.environ.get("PROXIMITY_FAR_THRESHOLD", "0.05"))
PROXIMITY_APPROACHING_THRESHOLD: float = float(os.environ.get("PROXIMITY_APPROACHING_THRESHOLD", "0.22"))
ARRIVAL_AREA_THRESHOLD: float = float(os.environ.get("ARRIVAL_AREA_THRESHOLD", "0.22"))

# ── Target detection reliability ──────────────────────────────
# The requested object is accepted from this confidence; every other class
# still needs CONFIDENCE_THRESHOLD (set equal to it to disable).
TARGET_CONFIDENCE_THRESHOLD: float = float(os.environ.get("TARGET_CONFIDENCE_THRESHOLD", "0.25"))
# Temporal smoothing: target "found" only after MIN_HITS matches in the last
# WINDOW frames; once found, gaps up to TARGET_HOLD_S keep the last box.
# TARGET_CONFIRM_MIN_HITS=1 and TARGET_HOLD_S=0 disable smoothing.
TARGET_CONFIRM_WINDOW: int = int(os.environ.get("TARGET_CONFIRM_WINDOW", "5"))
TARGET_CONFIRM_MIN_HITS: int = int(os.environ.get("TARGET_CONFIRM_MIN_HITS", "2"))
TARGET_HOLD_S: float = float(os.environ.get("TARGET_HOLD_S", "0.5"))

# ── Open-vocabulary fallback (roadmap R3, decisions.md D-13) ──
# Targets the closed-set model cannot detect (watch, slippers, …) are searched
# with this model, alongside the closed-set model (which still provides the
# obstacles).  EXP-06: YOLOE-26s text prompt found a worn watch in 99.7 % of
# frames (vs 76.4 % prompt-free, 18.3 % YOLO-World) with no false sightings.
OPEN_VOCAB_ENABLED: bool = os.environ.get("OPEN_VOCAB_ENABLED", "true").lower() in ("true", "1", "yes")
OPEN_VOCAB_MODEL_PATH: str = os.environ.get("OPEN_VOCAB_MODEL_PATH", "yoloe-26s-seg.pt")
# Same as TARGET_CONFIDENCE_THRESHOLD for now; EXP-06 suggests 0.10 would help
# objects lying flat, but false-positive evidence is only 2 scenes (D-13).
OPEN_VOCAB_CONFIDENCE_THRESHOLD: float = float(os.environ.get("OPEN_VOCAB_CONFIDENCE_THRESHOLD", "0.25"))


def _parse_prompt_map(spec: str) -> dict[str, list[str]]:
    """'keys=bunch of keys;key=bunch of keys|keys' → {'keys': ['bunch of keys'], 'key': ['bunch of keys', 'keys']}"""
    result: dict[str, list[str]] = {}
    for item in spec.split(";"):
        target, _, prompts = item.partition("=")
        words = [w.strip().lower() for w in prompts.split("|") if w.strip()]
        if target.strip() and words:
            result[target.strip().lower()] = words
    return result


# Words given to the open-vocabulary model for a target (default: the target
# itself). EXP-07: keys on a table 99.7 % found with "bunch of keys" vs 0 %
# with "keys" (decisions.md D-14). Speech still uses the user's word.
# Env format: "target=prompt1|prompt2;target2=prompt".
OPEN_VOCAB_PROMPTS: dict[str, list[str]] = _parse_prompt_map(
    os.environ.get("OPEN_VOCAB_PROMPTS", "keys=bunch of keys;key=bunch of keys")
)

# ── Session control ───────────────────────────────────────────
# Consecutive "reached" frames needed before the search ends as ARRIVED
# (filters out single-frame detection glitches).
ARRIVAL_CONFIRM_FRAMES: int = int(os.environ.get("ARRIVAL_CONFIRM_FRAMES", "5"))
# End the search if the target has not been seen for this many seconds
# (0 disables the timeout).
SEARCH_TIMEOUT_S: float = float(os.environ.get("SEARCH_TIMEOUT_S", "60"))
# Stop the camera loop after this many consecutive failed frame reads.
MAX_CONSECUTIVE_READ_FAILURES: int = int(os.environ.get("MAX_CONSECUTIVE_READ_FAILURES", "50"))

# ── Milestone 5 — Audio TTS Feedback ──────────────────────────
# TTS settings (TTS_*) moved to module1_audio/audio_feedback/tts_config.py
# together with the speech engine; the env-var names are unchanged.

# ── Milestone 5 — Obstacle Detection & Avoidance ──────────────
# Seconds the last seen target box is remembered for the obstacle check, so
# the person/table holding a flickering target is not reported as blocking.
OBSTACLE_TARGET_MEMORY_S: float = float(os.environ.get("OBSTACLE_TARGET_MEMORY_S", "2.0"))
# Classes that can HOLD the target (box contains the target) and are then
# not reported as blocking it.  Furniture is deliberately not included.
OBSTACLE_HOLDER_CLASSES: tuple[str, ...] = tuple(
    c.strip() for c in os.environ.get("OBSTACLE_HOLDER_CLASSES", "person").split(",") if c.strip()
)
OBSTACLE_DETECTION_ENABLED: bool =os.environ.get("OBSTACLE_DETECTION_ENABLED", "true").lower() in ("true", "1", "yes")
OBSTACLE_CORRIDOR_LEFT: float = float(os.environ.get("OBSTACLE_CORRIDOR_LEFT", "0.25"))
OBSTACLE_CORRIDOR_RIGHT: float = float(os.environ.get("OBSTACLE_CORRIDOR_RIGHT", "0.75"))
OBSTACLE_CORRIDOR_Y_MIN: float = float(os.environ.get("OBSTACLE_CORRIDOR_Y_MIN", "0.35"))
OBSTACLE_MIN_AREA_RATIO: float = float(os.environ.get("OBSTACLE_MIN_AREA_RATIO", "0.05"))
OBSTACLE_CLASSES: tuple[str, ...] = (
    "chair", "couch", "dining table", "bed", "bench",
    "person", "tv", "refrigerator", "toilet", "suitcase"
)
