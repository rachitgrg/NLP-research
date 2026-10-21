#!/usr/bin/env python3
# test_model.py
# ─────────────────────────────────────────────────────────────
# Quick test of the fine-tuned YOLOv8 model.
# No camera required — downloads a sample image and runs detection.
#
# Usage:
#   python test_model.py                    # test on a downloaded sample image
#   python test_model.py --image path.jpg   # test on your own image
#   python test_model.py --camera           # test on live webcam (press Q to quit)
#   python test_model.py --val             # run full val-set mAP evaluation
#
# Outputs saved to: module2_vision/training/test_output/
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("[ERROR] ultralytics not installed. Run: pip install ultralytics")

try:
    import cv2
except ImportError:
    sys.exit("[ERROR] opencv-python not installed. Run: pip install opencv-python")

# ── Paths ─────────────────────────────────────────────────────
TRAINING_DIR  = Path(__file__).resolve().parent
BEST_PT      = TRAINING_DIR / "runs" / "train" / "weights" / "best.pt"
OUTPUT_DIR   = TRAINING_DIR / "test_output"
DATASET_YAML = TRAINING_DIR / "dataset.yaml"

# ── 15 target class names (must match train.py order) ─────────
TARGET_CLASSES = [
    "person", "chair", "dining table", "couch", "bed",
    "bottle", "cup", "bowl", "laptop", "cell phone",
    "book", "tv", "backpack", "umbrella", "suitcase",
]

# Sample images to download if no --image is provided
# These are public COCO val images that contain our target classes
SAMPLE_IMAGES = [
    ("https://images.cocodataset.org/val2017/000000397133.jpg", "sample_kitchen.jpg"),
    ("https://images.cocodataset.org/val2017/000000037777.jpg", "sample_people.jpg"),
    ("https://images.cocodataset.org/val2017/000000252219.jpg", "sample_laptop.jpg"),
]


def load_model() -> YOLO:
    if not BEST_PT.exists():
        sys.exit(
            f"[ERROR] best.pt not found at {BEST_PT}\n"
            "Has training finished? Run: python train.py --resume"
        )
    print(f"Loading model: {BEST_PT}")
    model = YOLO(str(BEST_PT))
    print(f"Model loaded. Classes: {model.names}")
    return model


def test_on_image(model: YOLO, image_path: Path, conf: float = 0.25) -> None:
    """Run detection on a single image and save the annotated result."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"\nRunning detection on: {image_path}")
    t0 = time.perf_counter()
    results = model.predict(
        source=str(image_path),
        conf=conf,
        device="cuda",
        imgsz=640,
        verbose=False,
    )
    elapsed = (time.perf_counter() - t0) * 1000

    result = results[0]
    boxes = result.boxes

    # Print detections
    print(f"Inference time : {elapsed:.1f} ms  ({1000/elapsed:.0f} FPS equivalent)")
    print(f"Detections     : {len(boxes)}")

    if len(boxes) == 0:
        print("  No objects detected above confidence threshold.")
    else:
        print(f"  {'Class':20s}  {'Conf':>6s}  {'BBox (x1,y1,x2,y2)'}")
        print(f"  {'-'*20}  {'-'*6}  {'-'*30}")
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = TARGET_CLASSES[cls_id] if cls_id < len(TARGET_CLASSES) else f"cls_{cls_id}"
            conf_val = float(box.conf[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            print(f"  {cls_name:20s}  {conf_val:>6.3f}  ({x1},{y1}) → ({x2},{y2})")

    # Save annotated image
    out_path = OUTPUT_DIR / f"detected_{image_path.name}"
    annotated = result.plot()
    cv2.imwrite(str(out_path), annotated)
    print(f"\nAnnotated image saved → {out_path}")


def test_on_camera(model: YOLO, conf: float = 0.35) -> None:
    """Run live detection on webcam. Press Q to quit."""
    print("\nOpening webcam for live detection. Press Q to quit.")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("[ERROR] Cannot open webcam at index 0.")

    cv2.namedWindow("Fine-tuned YOLOv8 — 15-class test", cv2.WINDOW_NORMAL)

    fps_times: list[float] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()
        results = model.predict(
            source=frame,
            conf=conf,
            device="cuda",
            imgsz=640,
            verbose=False,
            stream=False,
        )
        t1 = time.perf_counter()
        fps_times.append(t1 - t0)
        if len(fps_times) > 30:
            fps_times.pop(0)
        fps = 1.0 / (sum(fps_times) / len(fps_times))

        annotated = results[0].plot()
        cv2.putText(
            annotated, f"FPS: {fps:.1f}  Model: best.pt (15-class fine-tuned)",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
        cv2.imshow("Fine-tuned YOLOv8 — 15-class test", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Camera test finished.")


def run_val_eval(model: YOLO) -> None:
    """Run full mAP evaluation on the 3,000-image val set."""
    if not DATASET_YAML.exists():
        sys.exit(f"[ERROR] dataset.yaml not found at {DATASET_YAML}")

    print(f"\nRunning mAP evaluation on val set ({DATASET_YAML})...")
    print("This will take ~5–10 minutes.\n")

    val_results = model.val(
        data=str(DATASET_YAML),
        imgsz=640,
        device="cuda",
        verbose=True,
    )

    box = val_results.box
    print("\n" + "=" * 55)
    print("VALIDATION RESULTS (fine-tuned best.pt)")
    print("=" * 55)
    print(f"  mAP@50       : {box.map50:.4f}  ({box.map50*100:.1f}%)")
    print(f"  mAP@50-95   : {box.map:.4f}  ({box.map*100:.1f}%)")
    print(f"  Precision    : {box.mp:.4f}")
    print(f"  Recall       : {box.mr:.4f}")

    if hasattr(box, "ap50") and box.ap50 is not None:
        print("\n  Per-class AP@50:")
        for i, cls in enumerate(TARGET_CLASSES):
            ap = float(box.ap50[i]) if i < len(box.ap50) else float("nan")
            bar = "█" * int(ap * 30)
            print(f"    [{i:>2d}] {cls:20s}  {ap:.3f}  {bar}")
    print("=" * 55)


def download_sample(url: str, filename: str) -> Path:
    out = OUTPUT_DIR / filename
    OUTPUT_DIR.mkdir(exist_ok=True)
    if out.exists():
        print(f"Sample image already exists: {out}")
        return out
    print(f"Downloading sample image → {filename}...")
    urllib.request.urlretrieve(url, out)
    print(f"Downloaded: {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test the fine-tuned YOLOv8 model on images or webcam."
    )
    parser.add_argument("--image", type=Path, default=None,
                        help="Path to a local image file to test on.")
    parser.add_argument("--camera", action="store_true",
                        help="Test on live webcam instead of an image.")
    parser.add_argument("--val", action="store_true",
                        help="Run full mAP evaluation on the val set (~5-10 min).")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Detection confidence threshold (default: 0.25).")
    args = parser.parse_args()

    model = load_model()

    if args.val:
        run_val_eval(model)

    elif args.camera:
        test_on_camera(model, conf=args.conf)

    elif args.image:
        if not args.image.exists():
            sys.exit(f"[ERROR] Image not found: {args.image}")
        test_on_image(model, args.image, conf=args.conf)

    else:
        # Default: download and test all sample images
        print("\nNo --image or --camera specified.")
        print("Testing on 3 sample COCO images (downloading if needed)...\n")
        for url, filename in SAMPLE_IMAGES:
            img_path = download_sample(url, filename)
            test_on_image(model, img_path, conf=args.conf)
            print()

    print("\nDone.")


if __name__ == "__main__":
    main()
