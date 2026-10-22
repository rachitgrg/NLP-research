# evaluation/multilingual_benchmark.py
# ─────────────────────────────────────────────────────────────
# Module 1 – EXP-08: offline multilingual speech recognition
# benchmark (roadmap R9; decides D-15).
#
# For every recording in the manifest and every backend:
#   audio → transcriber (speech → English) → spaCy parser → the app's
#   target normaliser → compare with the gold object (alternatives
#   separated by "|").
# The headline metric is OBJECT ACCURACY: did the assistant end up
# searching for the right thing?  Also reported: language-ID accuracy,
# latency, model load time.
#
# Backend specs:
#   whisper:<size>        Whisper, language auto-detected
#   whisper:<size>+hint   Whisper, told the recording's language
#   sarvam                Sarvam cloud API (paid; needs --allow-cloud)
#
# Output (never overwrites):
#   evaluation/runs/multilingual_<date_time>.csv          one row per clip × backend
#   evaluation/runs/multilingual_<date_time>_summary.csv  per backend × language
#   evaluation/runs/multilingual_<date_time>.json         settings, versions, git commit
#
# Usage (from module1_audio/):
#   python evaluation/multilingual_benchmark.py --backends whisper:small whisper:large-v3+hint
#   python evaluation/multilingual_benchmark.py --manifest <csv> --limit 5
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
MODULE_ROOT = EVAL_DIR.parent
sys.path.insert(0, str(MODULE_ROOT))

import speech.asr_config as asr_config  # noqa: E402  (offline flags first)

DEFAULT_MANIFEST = EVAL_DIR / "multilingual" / "recordings.csv"
RUNS_DIR = EVAL_DIR / "runs"
NORMALIZER_PATH = MODULE_ROOT.parent / "module2_vision" / "matching" / "target_normalizer.py"


def load_normalizer():
    """The vision module's normalize_target, loaded by file path (no package clash)."""
    spec = importlib.util.spec_from_file_location("m2_target_normalizer", NORMALIZER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.normalize_target


def parse_backend(spec: str) -> dict:
    """'whisper:large-v3+hint' → {'name': spec, 'backend': 'whisper', 'model_size': 'large-v3', 'hint': True}"""
    hint = spec.endswith("+hint")
    base = spec[: -len("+hint")] if hint else spec
    backend, _, size = base.partition(":")
    if backend not in ("whisper", "sarvam"):
        raise ValueError(f"unknown backend in {spec!r}")
    if backend == "whisper" and not size:
        raise ValueError(f"{spec!r}: give a Whisper size, e.g. whisper:large-v3")
    return {"name": spec, "backend": backend, "model_size": size or None, "hint": hint}


def base_language(code: str | None) -> str:
    """'hi-IN' → 'hi'; None → ''."""
    return (code or "").split("-")[0].lower()


def object_correct(predicted: str | None, gold: str, normalize) -> bool:
    """True if the normalised prediction equals any normalised gold alternative."""
    if not predicted:
        return False
    golds = {normalize(g) for g in gold.split("|") if g.strip()}
    return normalize(predicted) in golds


def summarise(rows: list[dict]) -> list[dict]:
    """Per backend × language (and 'all'): n, object accuracy, LID accuracy, latency."""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["backend"], r["language"])].append(r)
        groups[(r["backend"], "all")].append(r)
    out = []
    for (backend, language), items in sorted(groups.items()):
        lat = [float(r["latency_s"]) for r in items if not r["error"]]
        out.append({
            "backend": backend,
            "language": language,
            "n": len(items),
            "object_accuracy": round(sum(r["correct"] == "True" for r in items) / len(items), 4),
            "language_id_accuracy": round(
                sum(base_language(r["detected_language"]) == r["language"] for r in items) / len(items), 4),
            "errors": sum(bool(r["error"]) for r in items),
            "mean_latency_s": round(statistics.mean(lat), 4) if lat else "",
            "median_latency_s": round(statistics.median(lat), 4) if lat else "",
        })
    return out


def _make_transcriber(b: dict, language: str | None, allow_cloud: bool, device: str | None):
    from speech.transcriber_factory import create_transcriber

    if b["backend"] == "sarvam":
        if not allow_cloud:
            raise SystemExit("sarvam is a paid cloud API: add --allow-cloud to include it.")
        return create_transcriber("sarvam", offline=False)
    return create_transcriber("whisper", offline=asr_config.OFFLINE_MODE, model_size=b["model_size"],
                              language=language or "auto", device=device)


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=MODULE_ROOT).stdout.strip()
    except OSError:
        return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="EXP-08 multilingual offline ASR benchmark.")
    parser.add_argument("--backends", nargs="+", default=["whisper:small", "whisper:large-v3", "whisper:large-v3+hint"])
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--languages", nargs="*", help="Only these language codes")
    parser.add_argument("--limit", type=int, default=0, help="First N clips only (smoke test)")
    parser.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    parser.add_argument("--allow-cloud", action="store_true", help="Allow the paid Sarvam API")
    args = parser.parse_args()

    manifest = Path(args.manifest)
    with open(manifest, encoding="utf-8") as f:
        clips = list(csv.DictReader(f))
    if args.languages:
        clips = [c for c in clips if c["language"] in args.languages]
    if args.limit:
        clips = clips[: args.limit]
    if not clips:
        raise SystemExit(f"No recordings in {manifest}. Record them with evaluation/record_multilingual.py")

    from nlp.keyword_extractor import load_nlp_model
    from nlp.query_parser import parse_query

    nlp = load_nlp_model("en_core_web_sm")
    normalize = load_normalizer()
    backends = [parse_backend(s) for s in args.backends]
    stamp = time.strftime("%Y-%m-%d_%H%M%S")
    RUNS_DIR.mkdir(exist_ok=True)
    rows, load_times, devices = [], {}, {}

    for b in backends:
        # One model per language when hinted (the hint is fixed at load time).
        languages = sorted({c["language"] for c in clips}) if b["hint"] else [None]
        for language in languages:
            t0 = time.perf_counter()
            transcriber = _make_transcriber(b, language, args.allow_cloud, args.device)
            load_times[f"{b['name']}/{language or 'auto'}"] = round(time.perf_counter() - t0, 2)
            devices[b["name"]] = getattr(transcriber, "device", "cloud")
            todo = [c for c in clips if language is None or c["language"] == language]
            for c in todo:
                audio = manifest.parent / c["audio_file"] if not Path(c["audio_file"]).is_absolute() else Path(c["audio_file"])
                if not audio.exists():
                    audio = EVAL_DIR / c["audio_file"]
                t1 = time.perf_counter()
                text, detected, predicted, error = "", "", "", ""
                try:
                    res = transcriber.transcribe(str(audio))
                    text, detected = res["text"], res.get("language", "")
                    predicted = parse_query(nlp, text).get("object") or ""
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                latency = time.perf_counter() - t1
                ok = object_correct(predicted, c["object"], normalize)
                rows.append({
                    "audio_file": c["audio_file"], "language": c["language"], "speaker": c.get("speaker", ""),
                    "condition": c.get("condition", ""), "prompt_id": c.get("prompt_id", ""), "gold": c["object"],
                    "backend": b["name"], "text": text, "detected_language": detected,
                    "predicted_object": predicted, "normalized": normalize(predicted) if predicted else "",
                    "correct": str(ok), "latency_s": f"{latency:.4f}", "error": error,
                })
                print(f"[{b['name']:<22}] {c['audio_file']:<45} {'OK ' if ok else 'BAD'} "
                      f"{predicted!r:<16} {text[:60]!r}")
            del transcriber

    out = RUNS_DIR / f"multilingual_{stamp}.csv"
    with open(out, "x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarise(rows)
    with open(RUNS_DIR / f"multilingual_{stamp}_summary.csv", "x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    import faster_whisper

    meta = {
        "script": "module1_audio/evaluation/multilingual_benchmark.py",
        "date": stamp, "git_commit": _git_commit(), "manifest": str(manifest), "clips": len(clips),
        "backends": args.backends, "devices": devices, "load_times_s": load_times,
        "offline_mode": asr_config.OFFLINE_MODE, "beam_size": asr_config.ASR_BEAM_SIZE,
        "compute_type": asr_config.ASR_COMPUTE_TYPE or "auto (int8_float16 cuda / int8 cpu)",
        "faster_whisper": faster_whisper.__version__, "python": sys.version.split()[0],
        "parser": "spaCy en_core_web_sm + nlp/query_parser.py", "normalizer": "module2_vision/matching/target_normalizer.py",
    }
    (RUNS_DIR / f"multilingual_{stamp}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\n{'backend':<24}{'lang':<6}{'n':>4}{'object acc':>12}{'LID acc':>9}{'median s':>10}")
    for s in summary:
        print(f"{s['backend']:<24}{s['language']:<6}{s['n']:>4}{s['object_accuracy']:>12.1%}"
              f"{s['language_id_accuracy']:>9.1%}{s['median_latency_s'] or 0:>10.2f}")
    print(f"\nResults: {out}")


if __name__ == "__main__":
    main()
