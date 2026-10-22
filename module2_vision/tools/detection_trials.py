# tools/detection_trials.py
# ─────────────────────────────────────────────────────────────
# Module 3 – Controlled detection trials (measurement tool)
#
# Records one controlled scene from the webcam for a fixed time and
# reports how reliably the target is detected.  Each frame is run
# ONCE at a low threshold; both pipelines are then computed from the
# same frames, so the comparison is not affected by scene changes:
#
#   OLD : target present if confidence >= 0.40 in that frame
#   NEW : target accepted from TARGET_CONFIDENCE_THRESHOLD (0.25)
#         + TargetTracker smoothing (2-of-5 confirm, 0.5 s hold)
#
# No images are saved. One summary row per run (metrics + the settings
# used) is appended to tools/results/detection_trials_<YYYY-MM-DD>.csv.
# detection_trials.csv holds the 2026-10-20 runs and is never written.
#
# Usage (from module2_vision/):
#   python tools/detection_trials.py --scene "bottle alone"
#   python tools/detection_trials.py --scene "in front of face" --seconds 20
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parent.parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from config.config import (  # noqa: E402
    CONFIDENCE_THRESHOLD,
    TARGET_CONFIDENCE_THRESHOLD,
    TARGET_CONFIRM_MIN_HITS,
    TARGET_CONFIRM_WINDOW,
    TARGET_HOLD_S,
)
from matching.temporal_filter import TargetTracker  # noqa: E402

RESULTS_DIR = MODULE_ROOT / "tools" / "results"
RECORD_CONF = 0.05  # low threshold used while recording; filtering happens in analysis


@dataclass
class Sample:
    t: float            # seconds since recording start
    target_conf: float  # highest target confidence in the frame (0.0 = none)
    bbox: list[int] | None
    person: bool        # a person detected at >= CONFIDENCE_THRESHOLD


def _streak_stats(present: list[bool], times: list[float]) -> tuple[int, float]:
    """Return (number of present->absent flips, longest absent gap in seconds)."""
    flips = sum(1 for a, b in zip(present, present[1:]) if a and not b)
    longest, start = 0.0, None
    for p, t in zip(present + [True], times + [times[-1] if times else 0.0]):
        if not p and start is None:
            start = t
        elif p and start is not None:
            longest = max(longest, t - start)
            start = None
    return flips, longest


def analyze(
    samples: list[Sample],
    base_conf: float = CONFIDENCE_THRESHOLD,
    target_conf: float = TARGET_CONFIDENCE_THRESHOLD,
    tracker: TargetTracker | None = None,
) -> dict[str, float]:
    """Compute OLD vs NEW pipeline metrics from one recorded sequence."""
    if not samples:
        raise ValueError("no frames recorded")
    tracker = tracker or TargetTracker()
    times = [s.t for s in samples]

    old = [s.target_conf >= base_conf for s in samples]
    new = []
    for s in samples:
        raw_found = s.target_conf >= target_conf and s.bbox is not None
        match = {"target": "t", "found": raw_found, "confidence": s.target_conf if raw_found else None,
                 "bbox": s.bbox if raw_found else None, "matched_class": "t" if raw_found else None,
                 "reason": None}
        new.append(bool(tracker.update(match, s.t)["found"]))

    detected = [s.target_conf for s in samples if s.target_conf > 0]
    old_flips, old_gap = _streak_stats(old, times)
    new_flips, new_gap = _streak_stats(new, times)
    duration = max(times[-1] - times[0], 1e-6)
    return {
        "frames": len(samples),
        "fps": round(len(samples) / duration, 1) if len(samples) > 1 else 0.0,
        "person_frames_pct": round(100 * sum(s.person for s in samples) / len(samples), 1),
        "raw_rate_at_base_pct": round(100 * sum(old) / len(samples), 1),
        "raw_rate_at_target_pct": round(100 * sum(s.target_conf >= target_conf for s in samples) / len(samples), 1),
        "old_found_pct": round(100 * sum(old) / len(samples), 1),
        "new_found_pct": round(100 * sum(new) / len(samples), 1),
        "old_lost_events": old_flips,
        "new_lost_events": new_flips,
        "old_longest_gap_s": round(old_gap, 2),
        "new_longest_gap_s": round(new_gap, 2),
        "median_conf": round(statistics.median(detected), 3) if detected else 0.0,
        "min_conf": round(min(detected), 3) if detected else 0.0,
    }


def record(args: argparse.Namespace) -> list[Sample]:
    """Show a live preview, count down, then record args.seconds of frames."""
    import cv2
    from detection.yolo_detector import YOLODetector

    detector = YOLODetector(args.model, RECORD_CONF, args.imgsz, args.device)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera}")
    window = "Detection trial (Q = abort)"
    samples: list[Sample] = []
    try:
        t_start = time.monotonic()
        rec_start = None
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            now = time.monotonic()
            dets = detector.detect(frame)
            targets = [d for d in dets if d["class_name"].lower() == args.target]
            best = max(targets, key=lambda d: d["confidence"], default=None)

            if best is not None:
                x1, y1, x2, y2 = best["bbox"]
                colour = (0, 200, 0) if best["confidence"] >= TARGET_CONFIDENCE_THRESHOLD else (0, 165, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                cv2.putText(frame, f"{args.target} {best['confidence']:.2f}", (x1, max(15, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)

            if rec_start is None:
                remaining = args.countdown - (now - t_start)
                label = f"GET READY {remaining:.0f}"
                if remaining <= 0:
                    rec_start = now
                    print(f"Recording {args.seconds:.0f} s...")
            else:
                elapsed = now - rec_start
                samples.append(Sample(
                    t=elapsed,
                    target_conf=best["confidence"] if best else 0.0,
                    bbox=best["bbox"] if best else None,
                    person=any(d["class_name"] == "person" and d["confidence"] >= CONFIDENCE_THRESHOLD
                               for d in dets),
                ))
                label = f"REC {args.seconds - elapsed:.0f}s"
                if elapsed >= args.seconds:
                    break

            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.imshow(window, frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                print("Aborted.")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return samples


def results_file(fieldnames: list[str], results_dir: Path = RESULTS_DIR) -> Path:
    """Today's results file; a timestamped one if today's has other columns."""
    path = results_dir / f"detection_trials_{time.strftime('%Y-%m-%d')}.csv"
    if path.exists():
        with path.open(newline="", encoding="utf-8") as fh:
            if next(csv.reader(fh), None) != fieldnames:
                path = path.with_name(f"{path.stem}_{time.strftime('%H%M%S')}.csv")
    return path


def save_row(
    args: argparse.Namespace, metrics: dict[str, float], results_dir: Path = RESULTS_DIR
) -> Path:
    """Append one run (metrics + the settings that produced them) to a dated CSV."""
    row = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "scene": args.scene,
           "target": args.target, "model": Path(args.model).name, **metrics,
           "seconds": args.seconds, "imgsz": args.imgsz, "device": args.device,
           "base_threshold": CONFIDENCE_THRESHOLD,
           "target_threshold": TARGET_CONFIDENCE_THRESHOLD,
           "confirm_window": TARGET_CONFIRM_WINDOW,
           "confirm_min_hits": TARGET_CONFIRM_MIN_HITS, "hold_s": TARGET_HOLD_S}
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_file(list(row), results_dir)
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row))
        if new_file:
            writer.writeheader()
        writer.writerow(row)
    return path


def main() -> None:
    from config.config import CAMERA_INDEX, INFERENCE_DEVICE, INFERENCE_IMG_SIZE, YOLO_MODEL_PATH

    p = argparse.ArgumentParser(description="Record one controlled detection trial.")
    p.add_argument("--scene", required=True, help='Label, e.g. "bottle alone"')
    p.add_argument("--target", default="bottle")
    p.add_argument("--seconds", type=float, default=15.0)
    p.add_argument("--countdown", type=float, default=5.0)
    p.add_argument("--camera", type=int, default=CAMERA_INDEX)
    p.add_argument("--model", default=YOLO_MODEL_PATH)
    p.add_argument("--imgsz", type=int, default=INFERENCE_IMG_SIZE)
    p.add_argument("--device", default=INFERENCE_DEVICE)
    args = p.parse_args()
    args.target = args.target.lower().strip()

    samples = record(args)
    if len(samples) < 10:
        sys.exit("Too few frames recorded; nothing saved.")
    m = analyze(samples)
    path = save_row(args, m)

    print(f"\nScene: {args.scene}   ({m['frames']} frames, {m['fps']} FPS, person in {m['person_frames_pct']}% of frames)")
    print(f"  Raw detection rate  @{CONFIDENCE_THRESHOLD:.2f}: {m['raw_rate_at_base_pct']}%   "
          f"@{TARGET_CONFIDENCE_THRESHOLD:.2f}: {m['raw_rate_at_target_pct']}%")
    print(f"  Confidence of detected {args.target}: median {m['median_conf']}, min {m['min_conf']}")
    print(f"  {'':22s}{'OLD (0.40, per frame)':>24s}{'NEW (0.25 + smoothing)':>26s}")
    print(f"  {'Target reported found':22s}{m['old_found_pct']:>23}%{m['new_found_pct']:>25}%")
    print(f"  {'Lost events':22s}{m['old_lost_events']:>24}{m['new_lost_events']:>26}")
    print(f"  {'Longest gap (s)':22s}{m['old_longest_gap_s']:>24}{m['new_longest_gap_s']:>26}")
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
