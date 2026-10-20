# assistant.py
# ─────────────────────────────────────────────────────────────
# Module 3 – Assistive Navigation Voice Assistant
#
# Voice-first assistant designed for blind and visually impaired users:
#   1. Welcomes user via voice ("What object are you looking for?")
#   2. Records microphone audio and runs Module 1's HearingPipeline
#      (Sarvam AI / Whisper + spaCy query parser) to extract target.
#   3. Confirms target or alerts if class is unsupported.
#   4. Activates camera, runs YOLOv8 detection + spatial zoning +
#      obstacle avoidance corridor analysis.
#   5. Continuously speaks directional and obstacle guidance until
#      user reaches the target object.
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import argparse
import logging
import sys
import time

from audio_feedback.tts_speaker import SpeechPriority, TTSSpeaker
from config.config import (
    CAMERA_INDEX,
    CONFIDENCE_THRESHOLD,
    INFERENCE_DEVICE,
    INFERENCE_IMG_SIZE,
    YOLO_MODEL_PATH,
)
from detection.camera_detector import run_camera_loop
from detection.yolo_detector import YOLODetector
from guidance.obstacle_detector import ObstacleDetector
from guidance.spatial_locator import SpatialLocator
from matching.target_matcher import TargetMatcher
from matching.target_normalizer import is_supported_class, normalize_target
from navigation.navigation_controller import NavigationController

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("assistant")

ASSISTANT_BANNER = """\
╔══════════════════════════════════════════════════════════════╗
║        BLIND-ASSISTIVE OBJECT FINDING & NAVIGATION          ║
║                                                              ║
║  1. Voice Input (Module 1 STT + spaCy Query Parser)          ║
║  2. Live YOLO Detection & Target Matching                    ║
║  3. Spatial Directional Guidance (5 Horizontal Zones)        ║
║  4. Dynamic Obstacle Detection & Corridor Avoidance          ║
║  5. Real-Time Offline Speech Feedback (pyttsx3)              ║
╚══════════════════════════════════════════════════════════════╝
"""


def _get_target_from_speech(duration: int, speaker: TTSSpeaker) -> str | None:
    """Prompt user and extract target object using Module 1."""
    speaker.speak("Please say what object you are looking for.", priority=SpeechPriority.EMERGENCY, force=True)
    time.sleep(1.8)

    try:
        from pipeline.m3_pipeline import run_voice_pipeline
        from audio.microphone import record_audio, delete_temp_audio  # type: ignore
    except ImportError as exc:
        msg = "Voice module not available. Please type the target object."
        speaker.speak(msg)
        print(f"\n[INFO] {msg} ({exc})")
        return None

    audio_path = None
    try:
        print(f"\nListening for {duration} seconds... speak now.")
        audio_path = record_audio(duration=duration)
    except Exception as exc:
        print(f"[ERROR] Microphone capture failed: {exc}")
        return None

    try:
        target_info = run_voice_pipeline(audio_path)
    except Exception as exc:
        print(f"[ERROR] Voice pipeline failed: {exc}")
        return None
    finally:
        if audio_path:
            delete_temp_audio(audio_path)

    if not target_info:
        return None

    target = target_info.get("target", "")
    return target if target else None


def run_assistant(
    target: str | None = None,
    listen_voice: bool = True,
    duration: int = 5,
    camera_idx: int = CAMERA_INDEX,
    conf: float = CONFIDENCE_THRESHOLD,
    model_path: str = YOLO_MODEL_PATH,
    device: str = INFERENCE_DEVICE,
    use_mock_audio: bool = False,
) -> None:
    """Run full interactive assistive navigation session."""
    print(ASSISTANT_BANNER)
    speaker = TTSSpeaker(use_mock=use_mock_audio)

    raw_target = target

    # If no target specified via CLI, get from voice or keyboard prompt
    if not raw_target:
        if listen_voice:
            raw_target = _get_target_from_speech(duration, speaker)

        if not raw_target:
            speaker.speak("Please type the target object in the terminal.")
            try:
                raw_target = input("\nEnter object to find (e.g. bottle, phone, cup): ").strip()
            except (KeyboardInterrupt, EOFError):
                speaker.speak("Exiting. Goodbye.")
                speaker.stop()
                return

    if not raw_target:
        speaker.speak("No target object provided. Exiting.")
        speaker.stop()
        return

    normalized = normalize_target(raw_target)

    # Load YOLO detector first: whether a target is supported depends on
    # the loaded model's class vocabulary (15 classes for the fine-tuned
    # model, 80 for stock yolov8n.pt).
    print("\nLoading YOLO vision model...")
    try:
        detector = YOLODetector(
            model_path=model_path,
            confidence_threshold=conf,
            img_size=INFERENCE_IMG_SIZE,
            device=device,
        )
    except Exception as exc:
        logger.error("Failed to load YOLO model: %s", exc)
        speaker.speak("Vision model failed to load. Exiting.")
        speaker.stop()
        sys.exit(1)

    print("Model ready.")

    supported = is_supported_class(normalized, detector.class_names)

    print(f"\n  Requested Target : {raw_target!r}")
    print(f"  Normalized Class : {normalized!r}")
    print(f"  Supported Status : {'YES' if supported else 'NO (not in the loaded model classes)'}")

    if not supported:
        examples = ", ".join(sorted(detector.class_names.values())[:6])
        msg = f"{raw_target} is not a supported object for the vision model. Supported objects include {examples}."
        speaker.speak(msg, priority=SpeechPriority.EMERGENCY, force=True)
        print(f"\n[ALERT] {msg}")
        time.sleep(4.0)  # let the message finish before the speaker thread stops
        speaker.stop()
        return

    # Acknowledge target
    speaker.speak(f"Looking for {normalized}. Please hold camera steady and scan slowly.")

    # Instantiate navigation controller
    nav_controller = NavigationController(
        target_name=normalized,
        matcher=TargetMatcher(supported_classes=detector.class_names),
        locator=SpatialLocator(),
        obstacle_detector=ObstacleDetector(),
        speaker=speaker,
    )

    try:
        run_camera_loop(
            detector=detector,
            nav_controller=nav_controller,
            camera_index=camera_idx,
        )
    finally:
        speaker.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Blind-Assistive Navigation Assistant")
    parser.add_argument("--target", type=str, default=None, help="Target object name (skips voice input)")
    parser.add_argument("--duration", type=int, default=5, help="Voice recording duration in seconds")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX, help="Camera index")
    parser.add_argument("--conf", type=float, default=CONFIDENCE_THRESHOLD, help="YOLO confidence threshold")
    parser.add_argument("--device", type=str, default=INFERENCE_DEVICE, help="Inference device (cpu/cuda)")
    parser.add_argument("--mock-audio", action="store_true", help="Print audio cues to terminal instead of pyttsx3")
    args = parser.parse_args()

    run_assistant(
        target=args.target,
        listen_voice=args.target is None,
        duration=args.duration,
        camera_idx=args.camera,
        conf=args.conf,
        device=args.device,
        use_mock_audio=args.mock_audio,
    )


if __name__ == "__main__":
    main()
