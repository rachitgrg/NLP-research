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
#       python main.py --target watch        # not a closed-set class → open-vocabulary model (R3)
#
#   b) Voice input via Module 1 (full integration):
#       python main.py --voice
#       python main.py --voice --duration 8
#
#      --voice records a microphone utterance, sends it through
#      Module 1's HearingPipeline (offline Whisper + spaCy) to extract the
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
    REPLY_LANGUAGE,
    YOLO_MODEL_PATH,
    add_module1_to_path,
    environment_warning,
)
from detection.camera_detector import run_camera_loop
from detection.yolo_detector import YOLODetector
from matching.target_matcher import TargetMatcher
from detection.detector_factory import prepare_target_detector
from matching.target_normalizer import normalize_target

# TTS (audio_feedback) and the voice recorder live in module1_audio/.
add_module1_to_path()


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
    # ── M4/M5: Spatial Guidance, Obstacles & TTS ─────────────
    parser.add_argument(
        "--navigate",
        "--guidance",
        action="store_true",
        default=False,
        dest="navigate",
        help=(
            "M4/M5 mode: Enable real-time spatial directional guidance (far-left, "
            "left, center, right, far-right), proximity alerts, obstacle avoidance, "
            "and spoken voice feedback."
        ),
    )
    parser.add_argument(
        "--no-obstacles",
        action="store_true",
        default=False,
        help=(
            "Disable obstacle warnings. Use for desk testing with a webcam that "
            "faces you: otherwise you yourself are reported as 'Person ahead'."
        ),
    )
    parser.add_argument(
        "--no-tts",
        action="store_true",
        default=False,
        help="Disable spoken audio TTS output (print guidance text only).",
    )
    return parser.parse_args()


def _get_target_via_voice(duration: int) -> tuple[str | None, str | None]:
    """
    Record microphone audio, run Module 1's HearingPipeline, and return
    (extracted target object, spoken language code).

    Returns (None, None) if Module 1 is unavailable or extraction fails.
    """
    # Late import — Module 1 may not be installed in the M3 venv.
    try:
        from pipeline.m3_pipeline import run_voice_pipeline
    except ImportError as exc:
        logger.error("Cannot import m3_pipeline: %s", exc)
        return None, None

    # Module 1's microphone recorder lives in module1_audio/audio/microphone.py.
    # module1_audio/ is already on sys.path (add_module1_to_path / m3_pipeline).
    try:
        from audio.microphone import record_audio, delete_temp_audio  # type: ignore
    except ImportError:
        print(
            "\n[ERROR] Module 1 audio recorder not importable.\n"
            "Make sure Module 1 dependencies are installed and the module1_audio/ "
            "directory is present alongside module2_vision/."
        )
        return None, None

    audio_path = None
    try:
        print(f"\nRecording for {duration} second(s)... speak now.")
        audio_path = record_audio(duration=duration)
    except RuntimeError as exc:
        print(f"\n[ERROR] Microphone: {exc}")
        return None, None

    try:
        target_info = run_voice_pipeline(audio_path)
    except RuntimeError as exc:
        print(f"\n[ERROR] Module 1 pipeline failed: {exc}")
        return None, None
    finally:
        if audio_path:
            delete_temp_audio(audio_path)

    if target_info is None:
        return None, None

    target = target_info.get("target", "")
    language = target_info.get("language", "en")
    raw_text = target_info.get("raw_text", "")
    attributes = target_info.get("attributes", {})

    print(f"\n  Detected language : {language}")
    print(f"  Translated text   : {raw_text}")
    print(f"  Extracted target  : {target!r}")
    if attributes:
        print(f"  Attributes        : {attributes}  (preserved for M4+)")

    return (target if target else None), language


def main() -> None:
    args = parse_args()

    env_warning = environment_warning()
    if env_warning:
        print(f"\n{env_warning}\n")

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
    spoken_language: str | None = None

    if args.target:
        # Option A — text target supplied directly
        raw_target = args.target
        print(f"  Target     : {raw_target!r}  ->  normalized: {normalize_target(raw_target)!r}")

    elif args.voice:
        # Option B — voice input via Module 1
        print(f"  Mode       : VOICE INPUT  (duration: {args.duration}s)")
        print()
        raw_target, spoken_language = _get_target_via_voice(args.duration)
        if not raw_target:
            print(
                "\n[WARNING] Could not extract a target from voice input.\n"
                "         Falling back to M2 detection-only mode.\n"
            )
        else:
            print(f"  Target     : {raw_target!r}  ->  normalized: {normalize_target(raw_target)!r}")

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

    # Closed-set model if it knows the target; otherwise the open-vocabulary
    # model (R3) runs alongside it.  See detection/detector_factory.py.
    if raw_target:
        setup = prepare_target_detector(detector, normalize_target(raw_target), args.device)
        detector = setup.detector
        print(f"  Detection  : {setup.message}")
        if setup.mode == "unsupported":
            print("  WARNING: the target will show as UNSUPPORTED\n")

    # ── M3/M4/M5: initialise navigation controller or target matcher ──
    target_matcher = None
    nav_controller = None
    speaker = None

    if raw_target:
        if args.navigate:
            from audio_feedback.tts_speaker import TTSSpeaker, voice_languages
            from guidance.phrases import choose_reply_language
            from guidance.obstacle_detector import ObstacleDetector
            from guidance.spatial_locator import SpatialLocator
            from navigation.navigation_controller import NavigationController
            from matching.temporal_filter import TargetTracker

            # BUG FIX B1: NavigationController must receive the normalized target
            # (e.g. "cell phone", not "phone") because its internal matcher
            # compares against YOLO class_name values which are COCO canonical.
            # normalize_target() is already imported at the top of this file.
            nav_target = normalize_target(raw_target)

            # Replies in the spoken language when phrases + a voice exist (R9).
            reply_language = choose_reply_language(spoken_language, voice_languages(), REPLY_LANGUAGE)
            print(f"  Reply lang : {reply_language}")
            speaker = TTSSpeaker(enabled=not args.no_tts, language=reply_language)
            nav_controller = NavigationController(
                target_name=nav_target,
                matcher=TargetMatcher(supported_classes=detector.class_names),
                locator=SpatialLocator.for_target(nav_target),
                obstacle_detector=ObstacleDetector(enabled=not args.no_obstacles, language=reply_language),
                speaker=speaker,
                tracker=TargetTracker(),
                language=reply_language,
            )
        else:
            target_matcher = TargetMatcher(supported_classes=detector.class_names)

    # ── Start live camera + detection loop ─────────────────────
    try:
        run_camera_loop(
            detector=detector,
            target_matcher=target_matcher,
            raw_target=raw_target,
            nav_controller=nav_controller,
            camera_index=args.camera,   # BUG FIX B3: pass CLI --camera override
        )
    except RuntimeError as exc:
        logger.error("%s", exc)
        print(
            "\n[ERROR] Could not start camera. "
            "Make sure a webcam is connected and not used by another application.\n"
            "You can also try:\n"
            "  python main.py --camera 1\n"
        )
        sys.exit(1)
    finally:
        if speaker is not None:
            speaker.stop()


if __name__ == "__main__":
    main()
