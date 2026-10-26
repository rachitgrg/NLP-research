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

from config.config import add_module1_to_path

# TTS (audio_feedback) and the voice recorder live in module1_audio/.
add_module1_to_path()

from audio_feedback.tts_speaker import SpeechPriority, TTSSpeaker, voice_languages  # noqa: E402
from config.config import (  # noqa: E402
    CAMERA_INDEX,
    CONFIDENCE_THRESHOLD,
    INFERENCE_DEVICE,
    INFERENCE_IMG_SIZE,
    OFFLINE_MODE,
    REPLY_LANGUAGE,
    YOLO_MODEL_PATH,
    environment_warning,
)
from guidance.phrases import choose_reply_language, phrase  # noqa: E402
from navigation.session_log import StageTimer, append_session_row  # noqa: E402
from speech.asr_config import ASR_BACKEND, ASR_LANGUAGE, ASR_MODEL_SIZE  # noqa: E402  (module1_audio)
from detection.camera_detector import run_camera_loop
from detection.yolo_detector import YOLODetector
from guidance.obstacle_detector import ObstacleDetector
from guidance.spatial_locator import SpatialLocator
from matching.target_matcher import TargetMatcher
from detection.detector_factory import prepare_target_detector
from matching.target_normalizer import normalize_target
from matching.temporal_filter import TargetTracker
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
║  1. Voice Input (on-device Whisper + spaCy Query Parser)     ║
║  2. Live YOLO Detection & Target Matching                    ║
║  3. Spatial Directional Guidance (5 Horizontal Zones)        ║
║  4. Dynamic Obstacle Detection & Corridor Avoidance          ║
║  5. Offline Speech Feedback in the user's language           ║
╚══════════════════════════════════════════════════════════════╝
"""


def _get_target_from_speech(
    duration: int, speaker: TTSSpeaker, timer: StageTimer | None = None
) -> tuple[str | None, str | None]:
    """Prompt user and extract (target object, spoken language) using Module 1."""
    timer = timer or StageTimer()
    try:
        from pipeline.m3_pipeline import load_voice_pipeline, run_voice_pipeline
        from audio.microphone import record_audio, delete_temp_audio  # type: ignore
    except ImportError as exc:
        msg = "Voice module not available. Please type the target object."
        speaker.speak(msg)
        print(f"\n[INFO] {msg} ({exc})")
        return None, None

    # Load the speech model before asking, so the reply after the request
    # isn't delayed by it (Whisper large-v3 takes ~8 s to load).
    print("\nLoading speech model...")
    try:
        with timer.stage("asr_load_s"):
            load_voice_pipeline()
    except Exception as exc:
        print(f"[ERROR] Speech model failed to load: {exc}")
        return None, None

    speaker.speak(phrase("ask_target", speaker.language), priority=SpeechPriority.EMERGENCY, force=True)
    time.sleep(1.8)

    audio_path = None
    try:
        print(f"\nListening for {duration} seconds... speak now.")
        with timer.stage("record_s"):
            audio_path = record_audio(duration=duration)
    except Exception as exc:
        print(f"[ERROR] Microphone capture failed: {exc}")
        return None, None

    try:
        # Recognition only: the model was loaded above (asr_load_s).
        with timer.stage("asr_s"):
            target_info = run_voice_pipeline(audio_path)
    except Exception as exc:
        print(f"[ERROR] Voice pipeline failed: {exc}")
        return None, None
    finally:
        if audio_path:
            delete_temp_audio(audio_path)

    if not target_info:
        return None, None

    target = target_info.get("target", "")
    language = target_info.get("language")
    print(f"\n  Heard ({language}): {target_info.get('raw_text', '')!r}")
    return (target if target else None), language


def run_assistant(
    target: str | None = None,
    listen_voice: bool = True,
    duration: int = 5,
    camera_idx: int = CAMERA_INDEX,
    conf: float = CONFIDENCE_THRESHOLD,
    model_path: str = YOLO_MODEL_PATH,
    device: str = INFERENCE_DEVICE,
    use_mock_audio: bool = False,
    obstacles: bool = True,
    scenario: str = "",
) -> None:
    """Run full interactive assistive navigation session."""
    timer = StageTimer()
    print(ASSISTANT_BANNER)
    env_warning = environment_warning()
    if env_warning:
        print(f"{env_warning}\n")
    # Replies start in the configured speech language (ASR_LANGUAGE) and then
    # follow the language actually heard (REPLY_LANGUAGE, guidance/phrases.py).
    voices = voice_languages()
    speaker = TTSSpeaker(use_mock=use_mock_audio,
                         language=choose_reply_language(ASR_LANGUAGE, voices, REPLY_LANGUAGE))

    raw_target = target
    spoken_language: str | None = None

    # If no target specified via CLI, get from voice or keyboard prompt
    if not raw_target:
        if listen_voice:
            raw_target, spoken_language = _get_target_from_speech(duration, speaker, timer)

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

    reply_language = choose_reply_language(spoken_language, voices, REPLY_LANGUAGE)
    if reply_language != speaker.language:
        speaker.stop()
        speaker = TTSSpeaker(use_mock=use_mock_audio, language=reply_language)
    heard = (spoken_language or "en").split("-")[0]
    print(f"\n  Reply language   : {reply_language}"
          + ("" if reply_language == heard else f"  (no phrases/voice for {heard!r}; see REPLY_LANGUAGE)"))

    # Load YOLO detector first: whether a target is supported depends on
    # the loaded model's class vocabulary (15 classes for the fine-tuned
    # model, 80 for stock yolov8n.pt).
    print("\nLoading YOLO vision model...")
    try:
        with timer.stage("model_load_s"):
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

    # Closed-set model if it knows the target; otherwise the open-vocabulary
    # model (R3) runs alongside it.  See detection/detector_factory.py.
    closed_set_classes = dict(detector.class_names)
    with timer.stage("detector_setup_s"):
        setup = prepare_target_detector(detector, normalized, device)
    detector = setup.detector

    print(f"\n  Requested Target : {raw_target!r}")
    print(f"  Normalized Class : {normalized!r}")
    print(f"  Detection        : {setup.message}")

    if setup.mode == "unsupported":
        examples = ", ".join(sorted(closed_set_classes.values())[:6])
        msg = f"{raw_target} is not a supported object for the vision model. Supported objects include {examples}."
        speaker.speak(msg, priority=SpeechPriority.EMERGENCY, force=True)
        print(f"\n[ALERT] {msg}")
        time.sleep(4.0)  # let the message finish before the speaker thread stops
        speaker.stop()
        return

    # Acknowledge target
    speaker.speak(phrase("looking_for", reply_language, obj=normalized))

    # Instantiate navigation controller
    nav_controller = NavigationController(
        target_name=normalized,
        matcher=TargetMatcher(supported_classes=detector.class_names),
        locator=SpatialLocator.for_target(normalized),
        obstacle_detector=(ObstacleDetector(language=reply_language) if obstacles
                           else ObstacleDetector(enabled=False, language=reply_language)),
        speaker=speaker,
        tracker=TargetTracker(),
        language=reply_language,
    )

    try:
        run_camera_loop(
            detector=detector,
            nav_controller=nav_controller,
            camera_index=camera_idx,
        )
    finally:
        speaker.stop()
        row = {
            "scenario": scenario, "target": normalized, "detection_mode": setup.mode,
            "spoken_language": spoken_language or "", "reply_language": reply_language,
            "asr_backend": ASR_BACKEND if spoken_language else "typed",
            "asr_model": ASR_MODEL_SIZE if spoken_language and ASR_BACKEND == "whisper" else "",
            "device": device, "offline_mode": OFFLINE_MODE, "camera_index": camera_idx,
            "finish_reason": nav_controller.finish_reason or "stopped",
            **timer.values, **nav_controller.timings(),
        }
        path = append_session_row(row)
        print("\n  Session timings  : " + ", ".join(
            f"{k}={v}" for k, v in row.items() if k.endswith("_s") and v is not None))
        print(f"  Logged to        : {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Blind-Assistive Navigation Assistant")
    parser.add_argument("--target", type=str, default=None, help="Target object name (skips voice input)")
    parser.add_argument("--duration", type=int, default=5, help="Voice recording duration in seconds")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX, help="Camera index")
    parser.add_argument("--conf", type=float, default=CONFIDENCE_THRESHOLD, help="YOLO confidence threshold")
    parser.add_argument("--device", type=str, default=INFERENCE_DEVICE, help="Inference device (cpu/cuda)")
    parser.add_argument("--mock-audio", action="store_true", help="Print audio cues to terminal instead of pyttsx3")
    parser.add_argument("--no-obstacles", action="store_true",
                        help="Disable obstacle warnings (desk testing with a webcam that faces you)")
    parser.add_argument("--scenario", default="", help="Label stored in the session latency log (EXP-09)")
    args = parser.parse_args()

    run_assistant(
        target=args.target,
        listen_voice=args.target is None,
        duration=args.duration,
        camera_idx=args.camera,
        conf=args.conf,
        device=args.device,
        use_mock_audio=args.mock_audio,
        obstacles=not args.no_obstacles,
        scenario=args.scenario,
    )


if __name__ == "__main__":
    main()
