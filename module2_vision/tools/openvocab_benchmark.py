# tools/openvocab_benchmark.py
# ─────────────────────────────────────────────────────────────
# Module 2 – Open-vocabulary detector benchmark (roadmap R3, EXP-06)
#
# For each model: load time, time to set a target (includes the first
# text-encoder load for text-prompt models), model-only inference
# latency on a fixed image, and — for prompt-free models — which
# everyday object words are in the built-in vocabulary.
#
# The closed-set default (fine-tuned best.pt) is measured the same way
# as a reference.  One row per model is appended to
# tools/results/openvocab_benchmark_<date>.csv with the settings used.
#
# Usage (from module2_vision/):
#   python tools/openvocab_benchmark.py
#   python tools/openvocab_benchmark.py --models yoloe-26s-seg.pt --runs 200
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parent.parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from config.config import (  # noqa: E402
    INFERENCE_DEVICE,
    INFERENCE_IMG_SIZE,
    PROJECT_ROOT,
    YOLO_MODEL_PATH,
)

DATA_DIR = Path(os.environ.get("DATA_DIR", PROJECT_ROOT / "data"))
DEFAULT_IMAGE = DATA_DIR / "yolo_dataset" / "images" / "val" / "COCO_train2014_000000000714.jpg"
DEFAULT_MODELS = [
    YOLO_MODEL_PATH,
    "yoloe-26n-seg-pf.pt",
    "yoloe-26s-seg-pf.pt",
    "yoloe-26s-seg.pt",
    "yolov8s-worldv2.pt",
]
DEFAULT_WORDS = (
    "watch,wristwatch,key,keys,slipper,slippers,sandal,shoe,glasses,eyeglasses,sunglasses,"
    "wallet,remote control,earphones,headphones,charger,pen,medicine,pill bottle,bottle,cell phone"
)
RESULTS_DIR = MODULE_ROOT / "tools" / "results"


def _latency(detector, frame, runs: int, warmup: int = 10) -> list[float]:
    import torch

    for _ in range(warmup):
        detector.detect(frame)
    times = []
    for _ in range(runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        detector.detect(frame)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return times


def benchmark_model(model: str, frame, args) -> dict:
    from detection.open_vocab_detector import OpenVocabDetector, infer_kind
    from detection.yolo_detector import YOLODetector

    t0 = time.perf_counter()
    try:
        kind = infer_kind(model)
    except ValueError:
        kind = "closed-set"
    if kind == "closed-set":
        det = YOLODetector(model, args.conf, args.imgsz, args.device)
        load_s, set_s = time.perf_counter() - t0, 0.0
        vocab_hits = ""
    else:
        det = OpenVocabDetector(model, args.conf, args.imgsz, args.device, kind=kind)
        load_s = time.perf_counter() - t0
        words = [w.strip() for w in args.words.split(",") if w.strip()]
        vocab_hits = ";".join(w for w in words if det.supports(w)) if kind == "yoloe-pf" else "any (text prompt)"
        t1 = time.perf_counter()
        det.set_target(args.sanity_target)
        set_s = time.perf_counter() - t1

    found = [d["confidence"] for d in det.detect(frame) if d["class_name"] == args.sanity_target]
    times = _latency(det, frame, args.runs)
    med = statistics.median(times)
    return {
        "model": Path(model).name,
        "kind": kind,
        "load_s": round(load_s, 2),
        "set_target_s": round(set_s, 2),
        "median_ms": round(med, 2),
        "p90_ms": round(sorted(times)[int(0.9 * len(times)) - 1], 2),
        "fps_model_only": round(1000 / med, 1),
        f"{args.sanity_target}_confs": ";".join(f"{c:.2f}" for c in sorted(found, reverse=True)),
        "vocab_hits": vocab_hits,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark open-vocabulary detectors.")
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--image", default=str(DEFAULT_IMAGE))
    p.add_argument("--runs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=INFERENCE_IMG_SIZE)
    p.add_argument("--device", default=INFERENCE_DEVICE)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--sanity-target", default="bottle", help="object present in --image")
    p.add_argument("--words", default=DEFAULT_WORDS, help="comma-separated words for the vocabulary check")
    args = p.parse_args()

    import cv2
    import torch
    import ultralytics

    frame = cv2.imread(args.image)
    if frame is None:
        sys.exit(f"Cannot read image {args.image}")

    env = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": args.device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "runs": args.runs,
        "image": Path(args.image).name,
    }
    rows = []
    for model in args.models:
        print(f"\n=== {model} ===")
        row = {**env, **benchmark_model(model, frame, args)}
        rows.append(row)
        for k in ("kind", "load_s", "set_target_s", "median_ms", "p90_ms", "fps_model_only"):
            print(f"  {k:16s} {row[k]}")
        print(f"  {args.sanity_target} confs    {row[f'{args.sanity_target}_confs'] or '(none)'}")
        print(f"  vocab hits       {row['vocab_hits']}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"openvocab_benchmark_{time.strftime('%Y-%m-%d')}.csv"
    fields = list(rows[0])
    new = not out.exists()
    with out.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
