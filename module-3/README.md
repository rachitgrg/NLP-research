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
- **Corridor Safety**: Evaluates obstacles in the walking corridor (`chair`, `table`, `couch`, `person`) and generates detour advice (*"Chair on left. Step slightly right."*).
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
| Default model | YOLOv8n **fine-tuned on 15 indoor classes** (Module 2, `train.py`) |
| Weights file | `../module-2/runs/train/weights/best.pt` (local only, not in git) |
| Classes | person, chair, dining table, couch, bed, bottle, cup, bowl, laptop, cell phone, book, tv, backpack, umbrella, suitcase |
| Alternative | `--model yolov8n.pt` — stock COCO model, 80 classes (auto-downloaded) |
| Inference | `cuda` if the installed torch build sees a GPU, otherwise `cpu` (override with `--device` / `INFERENCE_DEVICE`) |

Whether a requested target is "supported" is decided from the **loaded model's own class list**,
so with the default 15-class model `clock` or `remote` are reported as unsupported.

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

### Vocabulary Limitation

The detector can only find the classes it was trained on (15 for the default fine-tuned
model, 80 for stock `yolov8n.pt`). Objects detectable by **neither** model
(returned as `unsupported_class`) — see `IMPLEMENTATION_ROADMAP.md` for the planned
open-vocabulary fallback:

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

**Recommended:** use the unified environment at the repository root (`.venv`, CUDA torch),
which contains both Module 1 (voice) and Module 3 (vision) dependencies — required for
`--voice` and `assistant.py`. See the root `README.md` → *Setup*.

```powershell
# From inside module-3/, using the root environment
..\.venv\Scripts\Activate.ps1
python main.py
```

The older module-local `module-3/venv` (CPU-only torch, no voice packages) still works for
vision-only runs and the test suite.

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
  Status   : UNSUPPORTED (not one of the loaded model's classes)
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
| `--model PATH` | fine-tuned `best.pt` | Path to YOLO weights |
| `--conf FLOAT` | `0.40` | Confidence threshold |
| `--imgsz INT` | `640` | Inference image size (pixels) |
| `--device DEVICE` | auto (`cuda` if available, else `cpu`) | Inference device: `cpu`, `cuda`, `mps` |
| `--camera INDEX` | `0` | Webcam device index |
| `--target OBJECT` | *(none)* | M3: object to search for (e.g. `bottle`, `phone`) |
| `--voice` | off | Get the target from the microphone via Module 1 |
| `--navigate` | off | Spoken spatial guidance, obstacles, arrival / timeout |
| `--no-tts` | off | Print guidance instead of speaking it |

**Stopping a session:** press **ENTER** in the terminal (works without window focus), **Q** in
the video window, or Ctrl+C. In `--navigate` mode the session also ends automatically when the
target stays "within reach" for `ARRIVAL_CONFIRM_FRAMES` frames, or when it has not been seen
for `SEARCH_TIMEOUT_S` seconds.

---

## Configuration

All default values live in [`config/config.py`](config/config.py):

| Config variable | Default | Env variable override |
|---|---|---|
| `YOLO_MODEL_PATH` | `../module-2/runs/train/weights/best.pt` | `YOLO_MODEL_PATH` |
| `CONFIDENCE_THRESHOLD` | `0.40` | `CONFIDENCE_THRESHOLD` |
| `INFERENCE_IMG_SIZE` | `640` | `INFERENCE_IMG_SIZE` |
| `INFERENCE_DEVICE` | auto (`cuda` if torch sees a GPU, else `cpu`) | `INFERENCE_DEVICE` |
| `CAMERA_INDEX` | `0` | `CAMERA_INDEX` |
| `LOG_EVERY_N_FRAMES` | `30` | `LOG_EVERY_N_FRAMES` |
| `TARGET_STATUS_COOLDOWN_S` | `2.0` | `TARGET_STATUS_COOLDOWN_S` |
| `ARRIVAL_CONFIRM_FRAMES` | `5` | `ARRIVAL_CONFIRM_FRAMES` |
| `SEARCH_TIMEOUT_S` | `60` (0 = off) | `SEARCH_TIMEOUT_S` |
| `MAX_CONSECUTIVE_READ_FAILURES` | `50` | `MAX_CONSECUTIVE_READ_FAILURES` |

---

## Running the Tests

```bash
# From inside module-3/ with the venv activated
pytest tests/ -v
```

The tests use **mocking** — no camera, no model download, and no GPU are required.

Run `pytest tests/ -q` for the current count. Besides the original suites, the tests cover the
Module 1 bridge (`test_m3_pipeline_bridge.py` — one test is skipped unless Module 1's
packages are installed, i.e. outside the root `.venv`), the camera loop's session control
(`test_camera_loop.py`, fake camera), speech-flood protection, arrival confirmation, the search
timeout, and model-vocabulary support checks.

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
