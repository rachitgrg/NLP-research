# Module 3 — Camera & YOLO Object Detection + Target Matching

> Part of the NLP Research project — Assistive navigation for visually impaired users.
> This module covers **Milestone 2** (camera + YOLO) and **Milestone 3** (target matching).

---

## What This Module Does

### Milestone 2 — Live YOLO Detection

Opens the device webcam, runs **YOLOv8n** inference on every frame, and displays annotated
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

### Milestone 3 — Target Matching

Connects the **requested object from Module 1** with the **objects detected by YOLO**.
Answers the single question: **"Is the object the user asked for currently visible?"**

```
User voice input (Module 1)
    ↓
HearingPipeline.run(audio)
    → result["object"] = "bottle"
    ↓
TargetNormalizer.normalize_target("bottle")
    → canonical COCO class name
    ↓
YOLODetector.detect(frame)
    → list of detections
    ↓
TargetMatcher.match("bottle", detections)
    ↓
{"target": "bottle", "found": True, "confidence": 0.86, "bbox": [...], ...}
```

### Milestone 4 — Spatial Localization & Proximity

Calculates horizontal zones and proximity ratios from target bounding boxes:
- **5 Horizontal Zones**: `far left`, `left`, `center`, `right`, `far right`.
- **Proximity Estimation**: `far`, `approaching`, `reached` (within arm's reach).

### Milestone 5 — Assistive Navigation, Obstacle Avoidance & Offline TTS

Combines all components into an autonomous voice assistant for visually impaired users:
- **Corridor Safety**: Evaluates obstacles in the walking corridor (`chair`, `table`, `couch`, `person`) and generates detour advice (*"Caution: chair ahead on your left. Step slightly right"*).
- **Offline TTS Feedback**: Non-blocking voice synthesis with priority overrides for emergency hazard warnings and arrival cues.
- **Continuous Guidance Loop**: Updates guidance as user moves until arrival (*"Bottle is directly in front of you. Stop and reach out"*).

---

## Folder Structure

```
module-3/
│
├── config/
│   ├── __init__.py
│   └── config.py              # ALL tuneable parameters (thresholds, corridors, TTS)
│
├── audio_feedback/            # Milestone 5: Voice feedback engine
│   ├── __init__.py
│   └── tts_speaker.py         # Asynchronous, debounced pyttsx3 speech synthesizer
│
├── guidance/                  # Milestone 4 & 5: Spatial math & safety
│   ├── __init__.py
│   ├── spatial_locator.py     # 5 horizontal zones + proximity + arrival math
│   └── obstacle_detector.py   # Corridor obstacle filtering & detour generation
│
├── navigation/                # Milestone 5: Finite State Machine
│   ├── __init__.py
│   ├── navigation_state.py    # NavigationState enum & NavigationStep dataclass
│   └── navigation_controller.py# Orchestrates target matching, hazard alerts & arrival
│
├── detection/
│   ├── __init__.py
│   ├── yolo_detector.py       # Reusable YOLODetector class (YOLO interface)
│   ├── camera_detector.py     # Live camera loop + orchestration (M2, M3, M4/M5)
│   └── visualizer.py          # Bounding-box drawing, guidelines & navigation overlays
│
├── matching/                  # Milestone 3 target matching
│   ├── __init__.py            # Exports SearchState
│   ├── target_normalizer.py   # Strip articles/possessives, alias mapping, COCO-80 set
│   ├── target_matcher.py      # TargetMatcher — core M3 component
│   └── search_state.py        # SearchState enum (SEARCHING / TARGET_FOUND)
│
├── pipeline/                  # Module 1 integration bridge
│   ├── __init__.py
│   └── m3_pipeline.py         # extract_target_from_module1() + run_voice_pipeline()
│
├── tests/
│   ├── test_config.py         # Config value type + env-override tests
│   ├── test_yolo_detector.py  # Detector output structure (mocked)
│   ├── test_visualizer.py     # Draw function tests
│   ├── test_target_matcher.py # Target normalizer + matcher tests
│   ├── test_spatial_locator.py# Milestone 4 spatial zone + proximity tests
│   ├── test_obstacle_detector.py # Milestone 5 obstacle detection & detour tests
│   ├── test_tts_speaker.py    # Milestone 5 TTS queue and debounce tests
│   ├── test_navigation_controller.py # State machine lifecycle tests
│   ├── test_assistive_pipeline.py # End-to-end 5-step assistive flow tests
│   └── test_m3_integration.py # Integration test suite
│
├── assistant.py               # Dedicated voice-first assistant runner for blind users
├── main.py                    # Multi-mode entry point (M2, M3, M4/M5 --navigate)
├── requirements.txt
├── .gitignore
└── README.md                  # This file
```

---

## Which YOLO Model Is Used?

**YOLOv8n** ("nano") via the [Ultralytics](https://github.com/ultralytics/ultralytics) SDK.

| Property | Value |
|---|---|
| Model | YOLOv8n (pretrained on COCO) |
| Weights file | `yolov8n.pt` (~6 MB) |
| Dataset | COCO (80 object categories) |
| Download | Automatic on first run |
| Inference | CPU (default) — also supports CUDA / MPS |

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

---

## M3 — Target Matching

### How Target Matching Works

1. **Normalization** (`target_normalizer.py`):
   - Strips leading articles and possessives: `"my bottle"` → `"bottle"`
   - Lowercases: `"BOTTLE"` → `"bottle"`
   - Applies spoken-word alias mapping: `"phone"` → `"cell phone"`
   - Checks COCO-80 support: `"watch"` → flagged as `unsupported_class`

2. **Matching** (`target_matcher.py`):
   - Compares the normalized target against every detection's `class_name` (case-insensitive)
   - If multiple instances of the same class are detected, **the highest-confidence one** is returned
   - Returns a consistent result dict

### Result Schema

**Target found:**
```python
{
    "target":        "bottle",
    "found":         True,
    "confidence":    0.86,
    "bbox":          [77, 162, 188, 477],
    "matched_class": "bottle",
    "reason":        None
}
```

**Target not found:**
```python
{
    "target":        "bottle",
    "found":         False,
    "confidence":    None,
    "bbox":          None,
    "matched_class": None,
    "reason":        None
}
```

**Unsupported target (not a COCO-80 class):**
```python
{
    "target":        "watch",
    "found":         False,
    "confidence":    None,
    "bbox":          None,
    "matched_class": None,
    "reason":        "unsupported_class"
}
```

### Alias Mapping (spoken word → COCO class)

| User says | Resolved COCO class |
|---|---|
| `phone`, `mobile`, `smartphone` | `cell phone` |
| `sofa` | `couch` |
| `television`, `monitor`, `screen` | `tv` |
| `fridge` | `refrigerator` |
| `plant`, `flower pot` | `potted plant` |
| `table` | `dining table` |
| `ball`, `football`, `basketball` | `sports ball` |
| `motorbike` | `motorcycle` |
| `bike` | `bicycle` |

### COCO-80 Limitation

YOLOv8n is pretrained on COCO and can **only** detect its 80 classes.
Objects **not** detectable by this model (returned as `unsupported_class`):

| User says | Why not supported |
|---|---|
| `watch` / `wristwatch` | COCO has `clock` (wall clocks), not wristwatches. Mapping would be misleading. |
| `key` / `keys` | No COCO class. |
| `pen` / `pencil` | No COCO class. |
| `slipper` / `shoe` | No COCO class. |
| `glasses` | No COCO class. |

> **Important**: `"watch"` is deliberately **not** mapped to `"clock"`. A wristwatch is not
> a wall clock. Reporting a false positive to a visually impaired user would be harmful.

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

### Step 3 — Run the pipeline

```bash
python main.py
```

---

## Running the Pipeline

### M2 mode — detect all objects

```powershell
# Default settings (yolov8n, conf=0.40, CPU, camera 0)
python main.py

# Use a bigger model
python main.py --model yolov8s.pt

# Raise the confidence threshold
python main.py --conf 0.60

# Use GPU
python main.py --device cuda

# Use camera index 1
python main.py --camera 1
```

### M3 mode — find a specific object

**Option A — text target (no Module 1 required):**
```powershell
# Search for a bottle
python main.py --target bottle

# Search for a phone (alias resolves to "cell phone")
python main.py --target phone

# Test with an unsupported COCO class
python main.py --target watch

# Combine M3 with M2 flags
python main.py --target bottle --conf 0.50 --camera 0
```

**Option B — voice input via Module 1 (full integration):**
```powershell
# Record 5s of voice, let Module 1 extract the target
python main.py --voice

# Record 8s
python main.py --voice --duration 8
```
> Requires Module 1 dependencies (spaCy, sarvam-ai, python-dotenv) and
> a valid `SARVAM_API_KEY` in `../module-1/.env`.

### M3 terminal output examples

**Target found:**
```
  Target   : bottle
  Status   : FOUND
  Confidence: 0.86
  BBox     : [77, 162, 188, 477]
```

**Target not found:**
```
  Target   : bottle
  Status   : NOT FOUND
```

**Unsupported class:**
```
  Target   : watch
  Status   : UNSUPPORTED (not a COCO-80 class — YOLOv8n cannot detect this)
```

### M3 visual overlay

When `--target` is supplied, the live video window shows:
- A **green banner** at the bottom: `TARGET: bottle | FOUND conf=0.86`
- A **red banner** when not found: `TARGET: bottle | NOT FOUND`
- An **orange banner** for unsupported: `TARGET: watch | UNSUPPORTED CLASS`
- The matched bounding box is **highlighted in green** with a cross-hair at its centre

---

## Available CLI Flags

| Flag | Default | Description |
|---|---|---|\
| `--model PATH` | `yolov8n.pt` | Path to YOLO weights |
| `--conf FLOAT` | `0.40` | Confidence threshold |
| `--imgsz INT` | `640` | Inference image size (pixels) |
| `--device DEVICE` | `cpu` | Inference device: `cpu`, `cuda`, `mps` |
| `--camera INDEX` | `0` | Webcam device index |
| `--target OBJECT` | *(none)* | M3: object to search for (e.g. `bottle`, `phone`) |

---

## Configuration

All default values live in [`config/config.py`](config/config.py):

| Config variable | Default | Env variable override |
|---|---|---|
| `YOLO_MODEL_PATH` | `"yolov8n.pt"` | `YOLO_MODEL_PATH` |
| `CONFIDENCE_THRESHOLD` | `0.40` | `CONFIDENCE_THRESHOLD` |
| `INFERENCE_IMG_SIZE` | `640` | `INFERENCE_IMG_SIZE` |
| `INFERENCE_DEVICE` | `"cpu"` | `INFERENCE_DEVICE` |
| `CAMERA_INDEX` | `0` | `CAMERA_INDEX` |
| `LOG_EVERY_N_FRAMES` | `30` | `LOG_EVERY_N_FRAMES` |
| `TARGET_STATUS_COOLDOWN_S` | `2.0` | `TARGET_STATUS_COOLDOWN_S` |

---

## Running the Tests

```bash
# From inside module-3/ with the venv activated
pytest tests/ -v
```

The tests use **mocking** — no camera, no model download, and no GPU are required.

**Test counts:**
- `test_config.py` — 11 tests (config defaults + env overrides)
- `test_yolo_detector.py` — 13 tests (detection output, edge cases)
- `test_visualizer.py` — 8 tests (draw functions + navigation overlay)
- `test_target_matcher.py` — 26 tests (normalization, matching, schema)
- `test_spatial_locator.py` — 13 tests (5 horizontal zones, proximity, arrival, clamping)
- `test_obstacle_detector.py` — 10 tests (corridor hazards, detours, filters)
- `test_tts_speaker.py` — 9 tests (queue, priority bypass, debouncing, lifecycle)
- `test_navigation_controller.py` — 8 tests (state machine transitions, guiding, obstacles, arrival)
- `test_assistive_pipeline.py` — 1 test (complete 5-step user journey simulation)
- `test_m3_integration.py` — 30 tests (mandatory spec cases + SearchState + Module 1 bridge + full pipeline)
- **Total: 129 tests (100% passing)**

---

## Milestone Scope

| Milestone | What it does | Status |
|---|---|:---:|
| **M1** | Multilingual Voice STT + spaCy Query Parser (Module 1) | **Completed** |
| **M2** | Camera + YOLO: detect all objects in the live frame | **Completed** |
| **M3** | Target Matching: "is the user's requested object currently visible?" | **Completed** |
| **M4** | Spatial localization: 5 horizontal zones + distance proximity estimation | **Completed** |
| **M5** | Assistive Navigation: real-time voice guidance, obstacle avoidance, arrival trigger | **Completed** |

---

## Limitations

- Only the 80 COCO object classes are detectable (pretrained weights).
- Inference runs on CPU by default — typically 5–15 FPS on a mid-range laptop.
  Passing `--device cuda` improves throughput if a CUDA GPU is available.
- The confidence threshold (`0.40`) may need tuning per environment.
- No object tracking — each frame is matched independently.
- `"watch"` (wristwatch) cannot be found; the model knows `clock` (wall/table clock) only.
