# Diagnostic: per-frame arrival state for one target (not part of the repo).
# Usage (from module2_vision): ..\.venv\Scripts\python <this> --target watch --seconds 25
import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from config.config import add_module1_to_path  # noqa: E402
add_module1_to_path()
import cv2  # noqa: E402

from config.config import CAMERA_BACKEND, CONFIDENCE_THRESHOLD, INFERENCE_DEVICE, INFERENCE_IMG_SIZE, YOLO_MODEL_PATH  # noqa: E402
from detection.camera_detector import _open_camera  # noqa: E402
from detection.detector_factory import prepare_target_detector  # noqa: E402
from detection.yolo_detector import YOLODetector  # noqa: E402
from guidance.obstacle_detector import ObstacleDetector  # noqa: E402
from guidance.spatial_locator import SpatialLocator  # noqa: E402
from matching.target_matcher import TargetMatcher  # noqa: E402
from matching.temporal_filter import TargetTracker  # noqa: E402
from navigation.navigation_controller import NavigationController  # noqa: E402


class SilentSpeaker:
    language = "en"
    def speak(self, *a, **k): return True
    def clear(self): pass
    def stop(self): pass
    def wait_until_idle(self, *a, **k): pass


ap = argparse.ArgumentParser()
ap.add_argument("--target", required=True)
ap.add_argument("--seconds", type=float, default=25)
ap.add_argument("--camera", type=int, default=0)
ap.add_argument("--out", required=True)
args = ap.parse_args()

det = YOLODetector(model_path=YOLO_MODEL_PATH, confidence_threshold=CONFIDENCE_THRESHOLD, img_size=INFERENCE_IMG_SIZE, device=INFERENCE_DEVICE)
setup = prepare_target_detector(det, args.target, INFERENCE_DEVICE)
print("mode:", setup.mode)
matcher = TargetMatcher(supported_classes=setup.detector.class_names)
nav = NavigationController(target_name=args.target, matcher=matcher, locator=SpatialLocator(),
                           obstacle_detector=ObstacleDetector(enabled=False), speaker=SilentSpeaker(),
                           tracker=TargetTracker())
cap = _open_camera(args.camera, CAMERA_BACKEND)
rows, t0 = [], time.monotonic()
while time.monotonic() - t0 < args.seconds:
    ok, frame = cap.read()
    if not ok:
        continue
    h, w = frame.shape[:2]
    dets = setup.detector.detect(frame)
    raw = matcher.match(args.target, dets)
    cand = [d for d in dets if d["class_name"] != "person"]
    step = nav.process_frame(dets, w, h)
    loc = step.spatial_location
    rows.append({
        "t": round(time.monotonic() - t0, 3),
        "classes": "|".join(f"{d['class_name']}:{d['confidence']:.2f}" for d in cand),
        "raw_found": raw.get("found"), "raw_conf": raw.get("confidence"),
        "smoothed_found": step.target_found,
        "area": round(loc.area_ratio, 3) if loc else "",
        "reached": loc.is_reached if loc else "",
        "streak": nav._arrival_streak, "state": step.state.name, "finish": nav.finish_reason or "",
    })
    for d in dets:
        x1, y1, x2, y2 = map(int, d["bbox"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f"{rows[-1]['state']} area={rows[-1]['area']} streak={rows[-1]['streak']}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("arrival probe", frame)
    cv2.waitKey(1)
    if nav.finish_reason:
        print("finished:", nav.finish_reason, "at", rows[-1]["t"])
        break
cap.release()
cv2.destroyAllWindows()
with open(args.out, "w", newline="", encoding="utf-8") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0]))
    wr.writeheader()
    wr.writerows(rows)
n = len(rows)
print(f"frames={n}  raw_found={sum(1 for r in rows if r['raw_found'])}  smoothed_found={sum(1 for r in rows if r['smoothed_found'])}"
      f"  reached={sum(1 for r in rows if r['reached'] is True)}  max_streak={max(r['streak'] for r in rows)}")
print("saved", args.out)
