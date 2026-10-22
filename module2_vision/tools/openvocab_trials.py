# tools/openvocab_trials.py
# ─────────────────────────────────────────────────────────────
# Module 2 – Open-vocabulary desk trials (roadmap R3, EXP-06)
#
# Records ONE controlled scene from the webcam into memory, then runs
# every model over the SAME frames, so models are compared on
# identical input (unlike separate live runs).  Frames are kept in RAM
# only and discarded afterwards; no images are written to disk.
#
# Metrics come from tools.detection_trials.analyze() (same definitions
# as the bottle trials, EXP-05), plus the raw detection rate at a lower
# threshold because open-vocabulary confidences tend to be lower.
# The closed-set default model is also run once to record how often a
# person is in view (context for face/occlusion scenes).
#
# One row per (scene, model) is appended to
# tools/results/openvocab_trials_<date>.csv with the settings used.
#
# Usage (from module2_vision/):
#   python tools/openvocab_trials.py --target watch --scene "watch on desk"
#   python tools/openvocab_trials.py --target keys --scene "no keys" --seconds 10
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import argparse
import csv
import itertools
import sys
import time
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parent.parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from config.config import (  # noqa: E402
    CAMERA_INDEX,
    CONFIDENCE_THRESHOLD,
    INFERENCE_DEVICE,
    INFERENCE_IMG_SIZE,
    TARGET_CONFIDENCE_THRESHOLD,
    TARGET_CONFIRM_MIN_HITS,
    TARGET_CONFIRM_WINDOW,
    TARGET_HOLD_S,
    YOLO_MODEL_PATH,
)
from tools.detection_trials import RECORD_CONF, Sample, analyze  # noqa: E402

RESULTS_DIR = MODULE_ROOT / "tools" / "results"
DEFAULT_MODELS = ["yoloe-26s-seg-pf.pt", "yoloe-26s-seg.pt", "yolov8s-worldv2.pt"]
LOW_THRESHOLD = 0.10


def record_frames(args: argparse.Namespace) -> tuple[list, list[float]]:
    """Live preview + countdown, then keep args.seconds of raw frames in memory."""
    import cv2

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera}")
    window = "Open-vocab trial (Q = abort)"
    frames, times = [], []
    try:
        t_start, rec_start = time.monotonic(), None
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            now = time.monotonic()
            shown = frame.copy()
            if rec_start is None:
                remaining = args.countdown - (now - t_start)
                label = f"GET READY {remaining:.0f}"
                if remaining <= 0:
                    rec_start = now
                    print(f"Recording {args.seconds:.0f} s (frames stay in memory only)...")
            else:
                elapsed = now - rec_start
                frames.append(frame)
                times.append(elapsed)
                label = f"REC {args.seconds - elapsed:.0f}s"
                if elapsed >= args.seconds:
                    break
            cv2.putText(shown, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.imshow(window, shown)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                print("Aborted.")
                frames, times = [], []
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return frames, times


def person_flags(frames: list, args: argparse.Namespace) -> list[bool]:
    """Per frame: is a person in view (closed-set default model, >= base threshold)?"""
    from detection.yolo_detector import YOLODetector

    det = YOLODetector(YOLO_MODEL_PATH, CONFIDENCE_THRESHOLD, args.imgsz, args.device)
    return [any(d["class_name"] == "person" for d in det.detect(f)) for f in frames]


def parse_prompt_sets(values: list[str] | None, target: str) -> list[list[str]]:
    """'--prompts watch wristwatch watch+wristwatch' → [['watch'], ['wristwatch'], ['watch', 'wristwatch']]."""
    if not values:
        return [[target]]
    sets = []
    for value in values:
        words = [w.strip().lower() for w in value.split("+") if w.strip()]
        if words:
            sets.append(words)
    return sets or [[target]]


def run_model(model: str, frames: list, times: list[float], persons: list[bool],
              args: argparse.Namespace, prompts: list[str] | None = None) -> tuple[list[Sample], float]:
    """Run one open-vocabulary model (with one prompt set) over the recorded frames."""
    from detection.open_vocab_detector import OpenVocabDetector

    prompts = prompts or [args.target]
    det = OpenVocabDetector(model, RECORD_CONF, args.imgsz, args.device)
    if not any(det.supports(p) for p in prompts):
        raise ValueError(f"{model}: none of {prompts} is in its vocabulary")
    det.set_target(args.target, prompts=prompts)
    det.detect(frames[0])  # warm-up
    samples = []
    t0 = time.perf_counter()
    for f, t, person in zip(frames, times, persons):
        dets = det.detect(f)
        best = max(dets, key=lambda d: d["confidence"], default=None)
        samples.append(Sample(t=t, target_conf=best["confidence"] if best else 0.0,
                              bbox=best["bbox"] if best else None, person=person))
    ms = (time.perf_counter() - t0) * 1000 / len(frames)
    return samples, ms


def results_file(prefix: str, fieldnames: list[str], results_dir: Path = RESULTS_DIR) -> Path:
    """
    Today's results file with these columns: the first of
    ``<prefix>_<date>.csv`` / ``<prefix>_<date>_<HHMMSS>.csv`` whose header
    matches, else a new file (never mixes columns, never overwrites).
    """
    base = results_dir / f"{prefix}_{time.strftime('%Y-%m-%d')}.csv"
    for candidate in [base, *sorted(results_dir.glob(f"{base.stem}_*.csv"))]:
        if not candidate.exists():
            return candidate
        with candidate.open(newline="", encoding="utf-8") as fh:
            if next(csv.reader(fh), None) == fieldnames:
                return candidate
    return base.with_name(f"{base.stem}_{time.strftime('%H%M%S')}.csv")


def save_rows(rows: list[dict], results_dir: Path = RESULTS_DIR) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_file("openvocab_trials", list(rows[0]), results_dir)
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        if new:
            w.writeheader()
        w.writerows(rows)
    return path


def main() -> None:
    p = argparse.ArgumentParser(description="Compare open-vocabulary models on one recorded scene.")
    p.add_argument("--target", required=True, help='e.g. "watch", "keys", "slippers"')
    p.add_argument("--scene", required=True, help='Label, e.g. "watch on desk"')
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--prompts", nargs="+", default=None,
                   help='prompt words to compare on the same frames; join words with "+" to combine '
                        'them, e.g. --prompts watch wristwatch watch+wristwatch (default: the target)')
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--countdown", type=float, default=5.0)
    p.add_argument("--camera", type=int, default=CAMERA_INDEX)
    p.add_argument("--imgsz", type=int, default=INFERENCE_IMG_SIZE)
    p.add_argument("--device", default=INFERENCE_DEVICE)
    args = p.parse_args()
    args.target = args.target.lower().strip()

    frames, times = record_frames(args)
    if len(frames) < 10:
        sys.exit("Too few frames recorded; nothing saved.")
    print(f"Recorded {len(frames)} frames. Running models...")
    persons = person_flags(frames, args)

    rows = []
    prompt_sets = parse_prompt_sets(args.prompts, args.target)
    for model, prompts in itertools.product(args.models, prompt_sets):
        try:
            samples, ms = run_model(model, frames, times, persons, args, prompts)
        except ValueError as exc:
            print(f"  skipped: {exc}")
            continue
        m = analyze(samples)
        low = round(100 * sum(s.target_conf >= LOW_THRESHOLD for s in samples) / len(samples), 1)
        rows.append({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "scene": args.scene,
            "target": args.target, "model": Path(model).name, "prompts": "+".join(prompts), **m,
            f"raw_rate_at_{LOW_THRESHOLD:.2f}_pct": low, "ms_per_frame": round(ms, 1),
            "seconds": args.seconds, "imgsz": args.imgsz, "device": args.device,
            "base_threshold": CONFIDENCE_THRESHOLD, "target_threshold": TARGET_CONFIDENCE_THRESHOLD,
            "confirm_window": TARGET_CONFIRM_WINDOW, "confirm_min_hits": TARGET_CONFIRM_MIN_HITS,
            "hold_s": TARGET_HOLD_S,
        })
    frames.clear()  # nothing is kept after the run

    if not rows:
        sys.exit("No model could search for this target; nothing saved.")
    path = save_rows(rows)

    print(f"\nScene: {args.scene}   target: {args.target}   "
          f"({rows[0]['frames']} frames, person in {rows[0]['person_frames_pct']}% of frames)")
    print(f"  {'model':22s}{'prompts':24s}{'raw@0.10':>9s}{'raw@0.25':>9s}{'raw@0.40':>9s}{'found NEW':>10s}"
          f"{'lost':>6s}{'median conf':>12s}{'ms/frame':>9s}")
    for r in rows:
        print(f"  {r['model']:22s}{r['prompts'][:23]:24s}"
              f"{r[f'raw_rate_at_{LOW_THRESHOLD:.2f}_pct']:>8}%{r['raw_rate_at_target_pct']:>8}%"
              f"{r['raw_rate_at_base_pct']:>8}%{r['new_found_pct']:>9}%{r['new_lost_events']:>6}"
              f"{r['median_conf']:>12}{r['ms_per_frame']:>9}")
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
