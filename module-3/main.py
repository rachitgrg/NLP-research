# main.py
# ─────────────────────────────────────────────────────────────
# Module 3 – Camera & YOLO Object Detection  (Milestones 2 & 3)
# Entry point — run the live YOLO camera pipeline.
#
# Milestone 2 (M2) — default mode, no target flag:
#   python main.py
#   python main.py --model yolov8s.pt --conf 0.5
#   python main.py --camera 1 --device cuda
#
# Milestone 3 (M3) — supply a target one of two ways:
#
#   a) Direct text target (existing behaviour, no Module 1 required):
#       python main.py --target bottle
#       python main.py --target "cell phone"
#       python main.py --target phone        # alias -> "cell phone"
#       python main.py --target watch        # unsupported COCO class
#
#   b) Voice input via Module 1 (full integration):
#       python main.py --voice
#       python main.py --voice --duration 8
#
#      --voice records a microphone utterance, sends it through
#      Module 1's HearingPipeline (Sarvam AI + spaCy) to extract the
#      target object, then runs the M3 camera+matching pipeline.
#
# All parameters default to the values in config/config.py.
# Command-line flags override config values for a single run only.
# ─────────────────────────────────────────────────────────────

import argparse
import logging
import sys

# ── Bootstrap logging before project imports ──────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
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
from matching.target_matcher import TargetMatcher
from matching.target_normalizer import is_supported_class, normalize_target


BANNER = """\
╔══════════════════════════════════════════════════════╗
║      MODULE 3 — CAMERA & YOLO OBJECT DETECTION      ║
║                                                      ║
║  Real-time YOLOv8 detection on live webcam feed.     ║
║  Milestones 2 & 3: Camera + YOLO + Target Matching.  ║
╚══════════════════════════════════════════════════════╝
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Module 3: Live YOLO object detection via webcam.\n"
            "M2 mode (default): detects all objects.\n"
            "M3 mode (--target or --voice): also matches a specific requested object."
        )
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
        help=f"Confidence threshold (0-1). Default: {CONFIDENCE_THRESHOLD}.",
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
    # ── M3 option A: direct text target ───────────────────────
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        metavar="OBJECT",
        help=(
            "M3: object to search for in the camera feed. "
            "Examples: --target bottle  |  --target phone  |  --target 'cell phone'. "
            "When omitted the pipeline runs in M2 detection-only mode."
        ),
    )
    # ── M3 option B: voice input via Module 1 ─────────────────
    parser.add_argument(
        "--voice",
        action="store_true",
        default=False,
        help=(
            "M3: record a microphone utterance and use Module 1 "
            "(HearingPipeline + Sarvam AI) to extract the target object. "
            "Requires Module 1 dependencies to be installed. "
            "Mutually exclusive with --target."
        ),
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=5,
        metavar="SECONDS",
        help="Recording duration in seconds when --voice is used. Default: 5.",
    )
    return parser.parse_args()


def _get_target_via_voice(duration: int) -> str | None:
    """
    Record microphone audio, run Module 1's HearingPipeline, and return
    the extracted target object string.

    Returns None if Module 1 is unavailable or extraction fails.
    """
    # Late import — Module 1 may not be installed in the M3 venv.
    try:
        from pipeline.m3_pipeline import run_voice_pipeline
    except ImportError as exc:
        logger.error("Cannot import m3_pipeline: %s", exc)
        return None

    # Module 1's microphone recorder lives in module-1/audio/microphone.py.
    # m3_pipeline._try_import_module1() has already added module-1/ to sys.path.
    try:
        from audio.microphone import record_audio, delete_temp_audio  # type: ignore
    except ImportError:
        print(
            "\n[ERROR] Module 1 audio recorder not importable.\n"
            "Make sure Module 1 dependencies are installed and the module-1/ "
            "directory is present alongside module-3/."
        )
        return None

    audio_path = None
    try:
        print(f"\nRecording for {duration} second(s)... speak now.")
        audio_path = record_audio(duration=duration)
    except RuntimeError as exc:
        print(f"\n[ERROR] Microphone: {exc}")
        return None

    try:
        target_info = run_voice_pipeline(audio_path)
    except RuntimeError as exc:
        print(f"\n[ERROR] Module 1 pipeline failed: {exc}")
        return None
    finally:
        if audio_path:
            delete_temp_audio(audio_path)

    if target_info is None:
        return None

    target = target_info.get("target", "")
    language = target_info.get("language", "en")
    raw_text = target_info.get("raw_text", "")
    attributes = target_info.get("attributes", {})

    print(f"\n  Detected language : {language}")
    print(f"  Translated text   : {raw_text}")
    print(f"  Extracted target  : {target!r}")
    if attributes:
        print(f"  Attributes        : {attributes}  (preserved for M4+)")

    return target if target else None


def main() -> None:
    args = parse_args()

    # ── Validate mutually exclusive flags ──────────────────────
    if args.target and args.voice:
        print(
            "\n[ERROR] --target and --voice are mutually exclusive.\n"
            "Use --target for a text target or --voice for microphone input.\n"
        )
        sys.exit(1)

    print(BANNER)
    print(f"  Model      : {args.model}")
    print(f"  Confidence : {args.conf}")
    print(f"  Image size : {args.imgsz}")
    print(f"  Device     : {args.device}")
    print(f"  Camera idx : {args.camera}")

    # ── Resolve target string ──────────────────────────────────
    raw_target: str | None = None

    if args.target:
        # Option A — text target supplied directly
        raw_target = args.target
        normalized = normalize_target(raw_target)
        supported  = is_supported_class(normalized)
        print(f"  Target     : {raw_target!r}  ->  normalized: {normalized!r}", end="")
        if not supported:
            print("  WARNING: NOT a COCO-80 class (will show as UNSUPPORTED)", end="")
        print()

    elif args.voice:
        # Option B — voice input via Module 1
        print(f"  Mode       : VOICE INPUT  (duration: {args.duration}s)")
        print()
        raw_target = _get_target_via_voice(args.duration)
        if not raw_target:
            print(
                "\n[WARNING] Could not extract a target from voice input.\n"
                "         Falling back to M2 detection-only mode.\n"
            )
        else:
            normalized = normalize_target(raw_target)
            supported  = is_supported_class(normalized)
            print(f"  Target     : {raw_target!r}  ->  normalized: {normalized!r}", end="")
            if not supported:
                print("  WARNING: NOT a COCO-80 class (will show as UNSUPPORTED)", end="")
            print()

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

    # ── M3: initialise target matcher (if target is known) ─────
    target_matcher = None
    if raw_target:
        target_matcher = TargetMatcher()

    # ── Start live camera + detection loop ─────────────────────
    try:
        run_camera_loop(detector, target_matcher, raw_target)
    except RuntimeError as exc:
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
