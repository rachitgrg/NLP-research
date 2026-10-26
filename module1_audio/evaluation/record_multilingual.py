# evaluation/record_multilingual.py
# ─────────────────────────────────────────────────────────────
# Module 1 – Record the multilingual test set for EXP-08
# (offline ASR benchmark, roadmap R9).
#
# Shows each request from multilingual/prompts.csv; the speaker says
# it NATURALLY in their own language (English loanwords such as
# "phone" or "bottle" are fine; the Hindi column is only a suggestion).
# Each 5 s clip is saved as
#     multilingual/audio/<language>_<speaker>_<condition>_<prompt>.wav
# and listed in multilingual/recordings.csv (the benchmark manifest).
# The audio folder is gitignored: recordings of other people stay local
# unless they agree to share them.
#
# Usage (from module1_audio/):
#   python evaluation/record_multilingual.py --language hi --speaker S01
#   python evaluation/record_multilingual.py --language ta --speaker S03 --condition noisy
#   python evaluation/record_multilingual.py --language hi --speaker S01 --only p11 p12
# Language codes are Whisper codes: en hi bn mr ta te gu kn ml pa ur …
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR.parent))

ML_DIR = EVAL_DIR / "multilingual"
PROMPTS = ML_DIR / "prompts.csv"
MANIFEST = ML_DIR / "recordings.csv"
AUDIO_DIR = ML_DIR / "audio"
MANIFEST_FIELDS = ["audio_file", "language", "speaker", "condition", "prompt_id", "object", "english"]


def load_prompts(path: Path = PROMPTS) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def clip_name(language: str, speaker: str, condition: str, prompt_id: str) -> str:
    return f"{language}_{speaker}_{condition}_{prompt_id}.wav"


def append_manifest(row: dict, path: Path = MANIFEST) -> None:
    """Add or replace the manifest row for row["audio_file"]."""
    rows = []
    if path.exists():
        with open(path, encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if r["audio_file"] != row["audio_file"]]
    rows.append(row)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Record the EXP-08 multilingual test set.")
    parser.add_argument("--language", required=True, help="Whisper language code of the speaker, e.g. hi")
    parser.add_argument("--speaker", required=True, help="Anonymous speaker id, e.g. S01 (no names)")
    parser.add_argument("--condition", default="quiet", help="quiet | noisy | …")
    parser.add_argument("--duration", type=int, default=5)
    parser.add_argument("--only", nargs="*", help="Prompt ids to (re-)record")
    args = parser.parse_args()

    from evaluation.record_samples import record_audio  # countdown + silence check

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    prompts = [p for p in load_prompts() if not args.only or p["prompt_id"] in args.only]
    print(f"Recording {len(prompts)} requests: language={args.language}, "
          f"speaker={args.speaker}, condition={args.condition}")
    print("Say each request naturally in your language. Mixing in English words is fine.\n")

    for i, p in enumerate(prompts, 1):
        name = clip_name(args.language, args.speaker, args.condition, p["prompt_id"])
        while True:
            print("-" * 50)
            print(f"{i}/{len(prompts)}  [{p['prompt_id']}]  {p['english']}")
            if args.language == "hi":
                print(f"         e.g. {p['hindi_suggestion']}")
            input("Press ENTER, then speak...")
            if record_audio(str(AUDIO_DIR / name), duration=args.duration):
                break
        append_manifest({
            "audio_file": f"multilingual/audio/{name}",
            "language": args.language,
            "speaker": args.speaker,
            "condition": args.condition,
            "prompt_id": p["prompt_id"],
            "object": p["object"],
            "english": p["english"],
        })
    print(f"\nDone. Manifest: {MANIFEST}")
    print(f"Audio:    {AUDIO_DIR}{os.sep}")


if __name__ == "__main__":
    main()
