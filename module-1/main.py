# main.py
# ─────────────────────────────────────────────────────────────
# Module 1 – Hearing & Understanding
# Entry point — run the complete voice pipeline interactively.
#
# Usage:
#   python main.py                   # default 5s recording
#   python main.py --duration 8      # record for 8 seconds
# ─────────────────────────────────────────────────────────────

# ── Load .env FIRST — before any project imports ──────────────
# This must happen before HearingPipeline / SarvamTranscriber are
# imported so that SARVAM_API_KEY is already in os.environ when
# those modules initialise.
import os
from pathlib import Path
try:
    from dotenv import load_dotenv
    # Anchor to this file's directory so it works regardless of CWD
    _env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(dotenv_path=_env_path, override=False)
except ImportError:
    pass  # python-dotenv not installed — rely on shell env vars
# ─────────────────────────────────────────────────────────────

import argparse
import json
import sys

from audio.microphone import record_audio, delete_temp_audio
from pipeline.hearing_pipeline import HearingPipeline


BANNER = """
========================================
MODULE 1 - HEARING & UNDERSTANDING
========================================
  Multilingual Voice → English NLP
  Powered by Sarvam AI Saaras v3
========================================
"""

DIVIDER = "─" * 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Module 1: Multilingual Voice → English NLP pipeline"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=5,
        help="Microphone recording duration in seconds (default: 5).",
    )
    return parser.parse_args()


def print_result(result: dict) -> None:
    """Pretty-print the pipeline output to the terminal."""
    print()
    print(DIVIDER)
    print(f"  Intent:")
    print(f"  {result['intent']}")
    print()
    print(f"  Object:")
    print(f"  {result['object'] if result['object'] else 'None'}")
    print()
    print(f"  Attributes:")
    print(f"  {result['attributes']}")
    print()
    print(f"  Keywords:")
    print(f"  {result['keywords']}")
    print()
    print(DIVIDER)
    print()


def main() -> None:
    args = parse_args()

    print(BANNER)
    print(f"  Duration : {args.duration} second(s) per recording")
    print()

    # Initialise the pipeline — this validates the Sarvam API key early
    # and loads the spaCy NLP model.
    print("Initialising pipeline...")
    try:
        pipeline = HearingPipeline()
    except EnvironmentError as exc:
        print(f"\n[ERROR] {exc}")
        sys.exit(1)
    print("Pipeline ready.\n")

    # ── Interactive loop ────────────────────────────────────────
    print("Press ENTER to start recording, or type 'q' then ENTER to quit.")

    while True:
        try:
            user_input = input("\n> ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            # Ctrl+C or Ctrl+D — exit gracefully
            print("\n\nGoodbye!")
            sys.exit(0)

        if user_input in ("q", "quit", "exit"):
            print("Goodbye!")
            break

        # Record audio to a temporary WAV file
        audio_path = None
        try:
            print("\nRecording...")
            audio_path = record_audio(duration=args.duration)
        except RuntimeError as exc:
            print(f"\n[ERROR] Microphone: {exc}")
            print("Press ENTER to try again, or 'q' to quit.")
            continue

        # Run the full pipeline
        try:
            result = pipeline.run(audio_path)
        except (RuntimeError, FileNotFoundError) as exc:
            print(f"\n[ERROR] {exc}")
            print("Press ENTER to try again, or 'q' to quit.")
            continue
        except Exception as exc:
            print(f"\n[ERROR] Unexpected failure: {exc}")
            continue
        finally:
            # Always clean up the temp audio file
            delete_temp_audio(audio_path)

        # Display results
        print_result(result)

        # Also print the raw JSON so it's easy to copy
        print("  JSON output:")
        print(" ", json.dumps(result, ensure_ascii=False, indent=4))
        print()
        print("Press ENTER to record again, or 'q' to quit.")


if __name__ == "__main__":
    main()
