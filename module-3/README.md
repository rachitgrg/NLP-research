# Module 3 — Camera & YOLO Object Detection

> Part of the NLP Research project.
> This module implements the live computer-vision pipeline for Milestone 2.

---

## What This Module Does

This module adds a real-time object-detection pipeline on top of Module 1's voice output.
It opens the device webcam, runs **YOLOv8** inference on every frame, and displays annotated
bounding boxes with class names and confidence scores in a live OpenCV window.

```
Webcam
    ↓
OpenCV (frame capture)
    ↓
YOLODetector.detect(frame)           ← reusable interface for future modules
    → list of detections (dicts)
    ↓
Visualizer (draw_detections)
    → bounding boxes + labels
    ↓
Live display window  (press Q to quit)
```

**Example terminal output:**

```
[Frame     30]  3 object(s) detected:
   Detected: person               | conf: 0.94 | bbox: [12, 45, 310, 420]
   Detected: bottle               | conf: 0.91 | bbox: [380, 200, 480, 430]
   Detected: chair                | conf: 0.87 | bbox: [150, 100, 580, 470]
```

---

## Folder Structure

```
module-3/
│
├── config/
│   ├── __init__.py
│   └── config.py              # ALL tuneable parameters live here
│
├── detection/
│   ├── __init__.py
│   ├── yolo_detector.py       # Reusable YOLODetector class (YOLO interface)
│   ├── camera_detector.py     # Live camera loop + orchestration
│   └── visualizer.py          # Bounding-box drawing utilities
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py            # pytest sys.path bootstrap
│   ├── test_config.py         # Config value type + env-override tests
│   ├── test_yolo_detector.py  # Detector output structure (mocked, no GPU needed)
│   └── test_visualizer.py     # Draw function tests
│
├── main.py                    # Entry point — run the live pipeline
├── requirements.txt           # Dependencies with install instructions
├── .gitignore
└── README.md                  # This file
```

---

## Which YOLO Model Is Used?

**YOLOv8n** ("nano") via the [Ultralytics](https://github.com/ultralytics/ultralytics) SDK.

| Property | Value |
|---|---|
| Model | YOLOv8n (pretrained on COCO 128 classes) |
| Weights file | `yolov8n.pt` (~6 MB) |
| Dataset | COCO (80 object categories) |
| Download | Automatic on first run |
| Inference | CPU (default) — also supports CUDA / MPS |

YOLOv8n is the smallest and fastest YOLOv8 variant — suitable for real-time inference on a
normal laptop CPU. Larger variants (`yolov8s`, `yolov8m`, `yolov8l`, `yolov8x`) can be
selected via `--model` or `config.py`.

---

## Detection Output Structure

Each detected object is represented as a plain Python dict:

```python
{
    "class_name": "bottle",       # str  — human-readable COCO class name
    "class_id":   39,             # int  — COCO class index
    "confidence": 0.91,           # float — 0.0–1.0, rounded to 4 decimal places
    "bbox":       [x1, y1, x2, y2]  # list[int] — pixel coordinates
}
```

This structure is intentionally simple so that **future modules** (target matching,
navigation, obstacle avoidance) can consume it without any YOLO-specific knowledge.

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `ultralytics` | latest | YOLOv8 model loading, inference, COCO class names |
| `opencv-python` | 5.0.0.93 | Webcam capture, frame display, drawing |
| `numpy` | ≥ 2.0 | Array operations for frame data |
| PyTorch | (auto-installed by ultralytics) | YOLO inference backend |

> **Python 3.13 note:**  `opencv-python` 5.0.0.93 is the first release with a stable
> `cp37-abi3` wheel that works on Python 3.13. Earlier versions (4.x) do not have 3.13
> wheels on PyPI.

---

## Setup

### Step 1 — Create and activate a virtual environment

```powershell
# From inside module-3/
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

> First install will download PyTorch (~200 MB CPU build) and Ultralytics automatically.

### Step 3 — Run the pipeline

```bash
python main.py
```

---

## Running the Pipeline

```powershell
# Default settings (yolov8n, conf=0.40, CPU, camera index 0)
python main.py

# Use a bigger model
python main.py --model yolov8s.pt

# Raise the confidence threshold
python main.py --conf 0.60

# Use GPU (if CUDA is available)
python main.py --device cuda

# Use camera index 1 (e.g. external webcam)
python main.py --camera 1

# Combine flags
python main.py --model yolov8s.pt --conf 0.50 --device cuda --camera 0
```

### Available CLI flags

| Flag | Default | Description |
|---|---|---|
| `--model PATH` | `yolov8n.pt` | Path to YOLO weights |
| `--conf FLOAT` | `0.40` | Confidence threshold |
| `--imgsz INT` | `640` | Inference image size (pixels) |
| `--device DEVICE` | `cpu` | Inference device: `cpu`, `cuda`, `mps` |
| `--camera INDEX` | `0` | Webcam device index |

---

## Configuration

All default values live in [`config/config.py`](config/config.py) and can also be
overridden via environment variables — useful for deployment without editing code.

| Config variable | Default | Env variable override |
|---|---|---|
| `YOLO_MODEL_PATH` | `"yolov8n.pt"` | `YOLO_MODEL_PATH` |
| `CONFIDENCE_THRESHOLD` | `0.40` | `CONFIDENCE_THRESHOLD` |
| `INFERENCE_IMG_SIZE` | `640` | `INFERENCE_IMG_SIZE` |
| `INFERENCE_DEVICE` | `"cpu"` | `INFERENCE_DEVICE` |
| `CAMERA_INDEX` | `0` | `CAMERA_INDEX` |
| `LOG_EVERY_N_FRAMES` | `30` | `LOG_EVERY_N_FRAMES` |

---

## Expected Output

When you run `python main.py` you should see:

1. **Terminal**: model loading progress, then a detection summary every 30 frames:
   ```
   ───────────────────────────────────────────────────────
     YOLO Live Detection running.
     Model confidence threshold : 0.40
     Press  Q  in the video window to quit.
   ───────────────────────────────────────────────────────

   [Frame     30]  2 object(s) detected:
      Detected: person               | conf: 0.94 | bbox: [10, 50, 300, 420]
      Detected: bottle               | conf: 0.91 | bbox: [380, 180, 480, 430]
   ```

2. **Live window**: Your webcam feed with coloured bounding boxes around detected objects.
   Each box has a filled label showing `class_name confidence` (e.g., `bottle 0.91`).
   A status bar at the top shows total object count and FPS.

3. **Exit**: Press **Q** in the video window (or close it) to quit cleanly.

---

## Running the Tests

```bash
# From inside module-3/ with the venv activated
pip install pytest
pytest tests/ -v
```

The tests use **mocking** — no camera, no model download, and no GPU are required.

---

## Limitations

- Only the 80 COCO object classes are detected (pretrained weights).
- Inference runs on CPU by default — typically 5–15 FPS on a mid-range laptop.
  Passing `--device cuda` dramatically improves throughput if a CUDA GPU is available.
- The confidence threshold (`0.40`) may need tuning per environment (lighting, distance).
- No tracking across frames yet — each frame is processed independently.

---

## What Comes in Milestone 3?

> ⚠️ The following are **NOT** implemented here. They are listed for planning only.

**Milestone 3 — Target Matching & Localization:**

1. **Connect Module 1 output → Module 3 detector** — the `object` field extracted by
   the NLP parser (e.g. `"bottle"`) is passed to the detection loop.
2. **Target matching** — after every frame, check if any detection's `class_name`
   matches the requested object.
3. **Target localization** — once matched, compute the bounding box centre to determine
   where in the frame the object is (left / centre / right, near / far).
4. **Voice feedback** — trigger a TTS response when the target is found
   (e.g. "Your bottle is on the left").
5. **COCO 2014 / RefCOCO integration** — decide how the pre-processed dataset from
   Module 2 feeds into fine-tuning YOLO or a grounding model.
