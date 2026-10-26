# Offline, Multilingual Voice-Driven Object Finder for Visually Impaired Users

An assistive prototype for blind and low-vision users. You ask for an object by voice, in **Hindi or English** ("Where is my bottle?" / "मेरी बोतल कहाँ है?"). The laptop camera finds it, and short spoken directions in your language ("Bottle, left." / "बोतल, बाईं ओर।") guide you to it until it is within reach. **Everything runs on the device, with no internet connection.**

> [!NOTE]
> **Works today (laptop prototype, tested 2026-10-27):** on-device speech recognition (Whisper large-v3) for Hindi and English; a fine-tuned 15-class YOLOv8n detector plus an open-vocabulary fallback (YOLOE-26s) for other objects such as a watch or keys; 5-zone spoken guidance, obstacle warnings, and an automatic stop on arrival; spoken replies in the user's language. With Airplane mode on, 14/14 spoken requests were understood and 10/10 test sessions reached the object.
>
> **Not yet:** a user study with visually impaired people, measured distance (arrival is judged by box size), colour/position words ("the blue cup on the left"), a watch lying flat on a table, objects held in the hand, and a phone or glasses version.

---

## Contents
1. [How it works](#1-how-it-works)
2. [Results](#2-results)
3. [Repository layout](#3-repository-layout)
4. [Setup](#4-setup)
5. [Running the assistant](#5-running-the-assistant)
6. [Configuration](#6-configuration)
7. [Tests](#7-tests)
8. [Reproducing the experiments](#8-reproducing-the-experiments)
9. [Research record](#9-research-record)
10. [Limitations](#10-limitations)
11. [Roadmap](#11-roadmap)
12. [Third-party models and libraries](#12-third-party-models-and-libraries)

---

## 1. How it works

```
 "मेरी बोतल कहाँ है?"
        │
 Microphone (5 s) ─► Whisper large-v3, on-device ─► "Where is my bottle?" ─► spaCy rule parser ─► target: bottle
                     (speech → English, language given)                                              │
                                                                                                     ▼
 Webcam ─► YOLOv8n fine-tuned, 15 classes ─────────────┐                        is the target one of the 15 classes?
       └─► YOLOE-26s text prompt (only for other objects, e.g. "watch") ─┴─► target matching + temporal smoothing
                                                                                     │
                                         5 image zones · arrival by box size · obstacle warnings
                                                                                     │
                                     Windows voice in the user's language ◄──────────┘   "बोतल, बाईं ओर।"
```

- **Speech:** [faster-whisper](https://github.com/SYSTRAN/faster-whisper) `large-v3` translates the request to English on the GPU (≈ 0.55 s per request when the language is given). A rule-based spaCy parser extracts the object, and a normaliser maps synonyms (phone → cell phone, spectacles → glasses, चाबी → key). The Sarvam cloud API remains available as an opt-in backend and served as the cloud baseline.
- **Detection:** the fine-tuned 15-class model handles common indoor objects (bottle, cup, bowl, laptop, cell phone, book, tv, backpack, umbrella, suitcase; plus person, chair, dining table, couch and bed, which are also used as obstacles). Any other target is passed as a text prompt to YOLOE-26s, which runs alongside it. A prompt map uses better wording where it helps (keys → "bunch of keys").
- **Guidance:** the target's position gives five left-to-right zones (far left … far right). The session ends when the object's box is large enough for 5 frames in a row (22 % of the frame by default, smaller for small objects: watch 3 %, keys 9 %). Obstacles in the walking corridor trigger warnings.
- **Offline by default:** `OFFLINE_MODE=true` blocks every model download and library network call, and `tools/offline_check.py` checks that all model files are present before a session.

## 2. Results

All numbers come from [`research/`](research/README.md). Before citing any of them, check [claims_and_limitations.md](research/paper_notes/claims_and_limitations.md).

| What | Result | Conditions |
|---|---|---|
| **Offline speech recognition** (EXP-08) | Whisper large-v3 extracted the right object from **90.0 % of Hindi** and **90.0–91.7 % of English** requests | 120 clips: 3 speakers × 20 requests × 2 languages, quiet room, laptop microphone |
| Smaller Whisper models (Hindi) | tiny 33 % · base 55 % · small 82 % · medium 90 % | same clips |
| **Offline vs cloud** | Sarvam (cloud) 95.8 %: going offline costs 5–6 points | same clips; consistent gap, p = 0.10 after correction |
| **End-to-end, network off** (EXP-09) | **14/14** requests understood; **10/10** sessions reached the object (bottle, phone, cup, worn watch, keys × Hindi/English) | one tester, one desk, Airplane mode on |
| **Open-vocabulary detection** (EXP-06/07) | worn watch 99.7 % of frames, slippers 99.7 %, keys on a table 99.7 % with the prompt "bunch of keys" (0 % with "keys") | one desk, one recording per scene |
| **Speed, RTX 3050 laptop GPU** | closed-set detector 8–9 ms/frame · YOLOE-26s 17–22 ms/frame · speech 0.55 s/request | model-only timings |
| **Speed, CPU only** (EXP-10) | detectors 23 / 74 ms/frame; Whisper large-v3 7.5 s/request (small 1.2 s) | Ryzen 5 7640HS |
| Fine-tuned detector (EXP-03) | mAP50 0.481, mAP50-95 0.331 | ⚠ optimistic: the COCO hold-out likely overlaps the pretraining data |

## 3. Repository layout

```
NLP-Research/
├── module1_audio/              # everything audio
│   ├── speech/                 #   on-device Whisper (whisper_stt.py), backend choice (transcriber_factory.py, asr_config.py)
│   ├── nlp/                    #   spaCy rule parser
│   ├── pipeline/               #   hearing_pipeline.py: audio → English text → target
│   ├── audio_feedback/         #   speech output (Windows SAPI/OneCore voices, tts_config.py)
│   ├── evaluation/             #   ASR benchmarks, multilingual recorder, runs/ (dated results)
│   ├── tests/
│   └── main.py                 #   voice-only entry point
├── module2_vision/             # everything vision + the assistant
│   ├── detection/              #   YOLO detector, open-vocabulary detector, detector_factory.py
│   ├── matching/               #   target normaliser, matcher, temporal smoothing
│   ├── guidance/               #   zones/arrival (spatial_locator.py), obstacles, phrases.py (en/hi)
│   ├── navigation/             #   navigation controller, per-session latency log
│   ├── pipeline/               #   bridge to module1_audio (m3_pipeline.py)
│   ├── config/config.py        #   all vision tunables (env-overridable)
│   ├── tools/                  #   offline_check.py, detection/open-vocab trials, benchmarks
│   ├── training/               #   15-class YOLOv8n fine-tune (train.py); runs/ is gitignored
│   ├── notebooks/              #   COCO + RefCOCO inspection, preprocessing, EDA
│   ├── weights/                #   pretrained and open-vocabulary weights (gitignored)
│   ├── main.py                 #   detection/navigation entry point (--target, --voice, --navigate)
│   └── assistant.py            #   voice-first assistant
├── research/                   # source of truth for the paper (log, decisions, experiments, claims)
├── data/                       # gitignored: raw COCO 2014 / RefCOCO, processed CSVs, yolo_dataset/
├── requirements.txt            # one environment for the whole project
├── .env.example                # variable names only (copy to .env)
└── CLAUDE.md                   # working rules for AI-assisted sessions
```

Module guides: [module1_audio](module1_audio/README.md) · [module2_vision](module2_vision/README.md) · [training](module2_vision/training/README.md) · [notebooks](module2_vision/notebooks/README.md)

## 4. Setup

Tested on **Windows 11, Python 3.13 (64-bit), NVIDIA RTX 3050 6 GB**. A GPU is strongly recommended: without one, the detectors still run, but Whisper large-v3 needs ~7.5 s per request.

### 4.1 Environment
```powershell
# from the repository root
python -m venv .venv
.\.venv\Scripts\python -m pip install torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m spacy download en_core_web_sm
.\.venv\Scripts\python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```
`.env` is needed only for the optional Sarvam cloud backend (`copy .env.example .env`, then add `SARVAM_API_KEY`). Never commit it.

### 4.2 Models (one-time download, then offline)
| Model | Size | Where it lives | How to get it |
|---|---|---|---|
| Whisper large-v3 (`Systran/faster-whisper-large-v3`) | 3.1 GB | Hugging Face cache | downloaded on the first run with `OFFLINE_MODE=false` |
| YOLOE-26s + MobileCLIP2-B text encoder | 31 MB + 254 MB | `module2_vision/weights/` | downloaded on the first open-vocabulary search with `OFFLINE_MODE=false` |
| Fine-tuned YOLOv8n, 15 classes (`best.pt`) | 6 MB | `module2_vision/training/runs/train/weights/` | produced by `training/train.py` (not in git). Without it, set `YOLO_MODEL_PATH=weights/yolov8n.pt` to use the stock 80-class model |

```powershell
cd module2_vision
$env:OFFLINE_MODE = "false"          # allow the one-time downloads
..\.venv\Scripts\python assistant.py --target watch --no-obstacles   # fetches YOLOE + text encoder
$env:OFFLINE_MODE = "true"
..\.venv\Scripts\python tools\offline_check.py   # must end with "Ready to run offline."
```
The Whisper model is fetched the same way on the first voice request with `OFFLINE_MODE=false`.

### 4.3 Hindi voice (for Hindi replies)
Install the Windows Hindi voice from an **administrator** PowerShell (the Settings → Speech download can stall):
```powershell
Add-WindowsCapability -Online -Name "Language.TextToSpeech~~~hi-IN~0.0.1.0"
```
`offline_check.py` should then list reply voices `en, hi`. Without a Hindi voice, replies fall back to English.

## 5. Running the assistant

Run everything from inside `module2_vision/`.

```powershell
cd module2_vision

# Voice-first assistant: asks what you are looking for, listens 5 s, then guides you
$env:ASR_LANGUAGE = "hi"     # or "en"; giving the language makes recognition ~40 % faster
..\.venv\Scripts\python assistant.py --no-obstacles     # --no-obstacles for a desk test (webcam faces you)

# Typed target, spoken guidance
..\.venv\Scripts\python main.py --target bottle --navigate --no-obstacles

# Voice only (no camera): speech → parsed request
cd ..\module1_audio; ..\.venv\Scripts\python main.py
```

- A session **ends by itself** when the object is reached, or after 60 s without seeing it. To stop early, press **Q** in the video window or **ENTER** in the terminal.
- **Camera direction:** "left/right" are in image space, so the camera should face the way the user faces (a chest or phone camera). A laptop webcam facing you works for desk tests with `--no-obstacles`.
- **Timings:** `assistant.py --scenario <name>` logs per-stage times (speech-model load, recording, recognition, first sighting, arrival) to `tools/results/session_latency_<date>.csv`.

## 6. Configuration

Every tunable can be overridden with an environment variable (PowerShell: `$env:NAME = "value"`).

| Variable | Default | Meaning |
|---|---|---|
| `OFFLINE_MODE` | `true` | block model downloads and library network calls |
| `ASR_BACKEND` | `whisper` | `whisper` (on-device) or `sarvam` (cloud, needs `.env`) |
| `ASR_MODEL_SIZE` | `large-v3` | Whisper size (`small`/`medium` for weaker hardware) |
| `ASR_LANGUAGE` | `auto` | `hi`, `en`, … (recommended: set it) |
| `REPLY_LANGUAGE` | `auto` | language of spoken replies (follows the request when a voice exists) |
| `YOLO_MODEL_PATH` | `training/runs/train/weights/best.pt` | closed-set detector |
| `OPEN_VOCAB_ENABLED` | on | open-vocabulary fallback for other objects |
| `OPEN_VOCAB_PROMPTS` | `keys=bunch of keys;key=bunch of keys` | words given to the open-vocabulary model |
| `ARRIVAL_AREA_THRESHOLD` / `ARRIVAL_AREA_BY_TARGET` | `0.22` / `watch=0.03;keys=0.09;key=0.09` | box area that counts as "within reach" |
| `SEARCH_TIMEOUT_S` | `60` | stop after this long without seeing the target |
| `DATA_DIR` | `<repo>/data` | datasets for training and notebooks |

All values: `module2_vision/config/config.py`, `module1_audio/speech/asr_config.py`, `module1_audio/audio_feedback/tts_config.py`.

## 7. Tests

```powershell
cd module1_audio;  ..\.venv\Scripts\python -m pytest tests -q   # 126 tests
cd ..\module2_vision; ..\.venv\Scripts\python -m pytest tests -q   # 220 tests
```
The tests mock the camera, the models and the voice, and a `no_network` fixture fails any test that opens a connection. Hardware behaviour needs a manual run (section 5).

## 8. Reproducing the experiments

| Experiment | Command (from the module folder) |
|---|---|
| EXP-08 multilingual ASR | `evaluation\record_multilingual.py --language hi --speaker S01`, then `evaluation\multilingual_benchmark.py --backends whisper:large-v3+hint` (run one backend per process; `sarvam --allow-cloud` uses paid credit) |
| EXP-06/07 open-vocabulary | `tools\openvocab_benchmark.py`; `tools\openvocab_trials.py --target watch --scene "<name>" [--prompts a b]` |
| EXP-05 detection trials | `tools\detection_trials.py --scene "<name>"` |
| EXP-09 offline sessions | scenario sheet: [research/experiments/2026-10-27_offline_end_to_end](research/experiments/2026-10-27_offline_end_to_end/README.md) |
| EXP-03 training | `training\train.py` (overwrites `runs/train/`) |

Every script writes dated result files and never overwrites earlier runs. Full environment and dataset steps: [research/REPRODUCIBILITY.md](research/REPRODUCIBILITY.md).

## 9. Research record

[`research/`](research/README.md) is the single source of truth for the paper:
[LOG](research/LOG.md) (what happened, when) · [decisions](research/decisions.md) (D-01 … D-20) · [experiments](research/experiments/) (EXP-01 … EXP-10, each with script, settings, data and caveats) · [claims & limitations](research/paper_notes/claims_and_limitations.md) · [paper outline](research/paper_notes/outline.md) · [roadmap](research/paper_notes/IMPLEMENTATION_ROADMAP.md) · [mentor summary](research/paper_notes/MENTOR_SUMMARY_2026-10-27.md)

## 10. Limitations

- **No user study yet.** All tests were run by the developer at a desk; nothing has been measured with visually impaired users.
- **No distance measurement.** "Within reach" is a box-size rule, and left/right are image directions.
- **Speech evaluation:** Hindi and English only, 3 speakers, quiet room, one microphone.
- **Detection gaps:** objects held in the hand aren't found; a watch lying flat on a table is seen only at low confidence; only watch, keys and slippers were measured for the open-vocabulary path.
- **Hardware:** real-time speech needs the GPU; the phone/glasses version is future work.
- **Detector accuracy** (mAP50 0.481) is optimistic until it is re-measured on COCO val2017.
- **Attributes** such as colour and position are parsed but not yet used to choose between objects.

## 11. Roadmap

- **R10 (proposed): a watch on a table.** A test set first (several watches, surfaces, camera angles), then a downward camera, a lower watch threshold, higher resolution, or "show it your watch once" (an example image as the prompt).
- **R2:** a clean detector baseline on COCO val2017 (fine-tuned vs stock).
- **R6:** use colour and position words to choose between objects (evaluated on RefCOCO).
- **R7:** scripted-room trials and a user study; relative distance cues.
- **Later:** a phone or glasses version with a smaller speech model or an accelerator.

Details: [IMPLEMENTATION_ROADMAP.md](research/paper_notes/IMPLEMENTATION_ROADMAP.md).

## 12. Third-party models and libraries

- [Whisper](https://github.com/openai/whisper) via [faster-whisper](https://github.com/SYSTRAN/faster-whisper) / CTranslate2 (MIT)
- [Ultralytics](https://github.com/ultralytics/ultralytics) YOLOv8n, YOLOE-26 (AGPL-3.0) and the [ultralytics/CLIP](https://github.com/ultralytics/CLIP) fork; MobileCLIP2 text encoder
- [spaCy](https://spacy.io/) `en_core_web_sm` (MIT)
- Datasets: [COCO 2014](https://cocodataset.org/) and [RefCOCO](https://github.com/lichengunc/refer)
- Optional cloud baseline: [Sarvam AI](https://www.sarvam.ai/) Saaras v3
