# Multimodal Audio-Visual Grounding Framework for Real-Time Assistive Navigation

A research project building an assistive object finder for blind and low-vision users: the user says what they are looking for, the camera finds it, and spoken directions guide them to it.

> [!NOTE]
> **Implemented:** multilingual voice request → English text → rule-based parser → real-time 15-class YOLOv8n detection, with an open-vocabulary fallback (YOLOE) for other objects such as a watch or slippers → 5-zone spoken direction guidance with obstacle warnings and arrival detection.
> **Not yet implemented:** free-text visual grounding (e.g. "the red bottle on the left"), reliable detection of objects held in the hand (not found in desk tests), and metric distance estimation.
> Verified status: [research/PROJECT_AUDIT.md](research/PROJECT_AUDIT.md). Next steps: [research/paper_notes/IMPLEMENTATION_ROADMAP.md](research/paper_notes/IMPLEMENTATION_ROADMAP.md).

---

## 1. Repository layout

```
NLP-Research/
├── module1_audio/      # all sound: mic, Sarvam STT, spaCy parser, ASR benchmark, speech output (TTS)
├── module2_vision/     # all vision: detection, matching, guidance, navigation, assistant entry points
│   ├── training/       #   YOLOv8n 15-class fine-tuning (train.py), runs/ (gitignored)
│   ├── notebooks/      #   COCO + RefCOCO inspection, preprocessing, EDA
│   └── weights/        #   pretrained weights (gitignored)
├── data/               # gitignored: raw/ (COCO 2014, RefCOCO), processed/ (CSVs), yolo_dataset/
├── research/           # source of truth for the paper: log, decisions, dataset, experiments, results
├── .env.example        # variable names (copy to .env)
├── requirements.txt    # one environment for everything
└── CLAUDE.md           # working rules for AI-assisted sessions
```

Module guides: [module1_audio/README.md](module1_audio/README.md) · [module2_vision/README.md](module2_vision/README.md) · [training](module2_vision/training/README.md) · [notebooks](module2_vision/notebooks/README.md)

## 2. Architecture

`Microphone → Sarvam Saaras v3 (speech → English) → spaCy parser (intent / object / attributes) → target check against the model's classes → webcam → YOLOv8n (15 classes) → target matching + temporal smoothing → 5-zone spatial locator + corridor obstacles → speech output → arrival / timeout / cancel`

Diagrams: [Stage-1 architecture](research/figures/fig_01_stage1_architecture.png) (voice part only) · [dataset pipeline](research/figures/fig_13_dataset_pipeline.png)

## 3. Research record

Everything for the paper lives in [`research/`](research/README.md):
[LOG](research/LOG.md) (what happened, when) · [decisions](research/decisions.md) · [dataset](research/dataset/README.md) · [experiments](research/experiments/) · [results manifest](research/results/MANIFEST.md) · [claims & limitations](research/paper_notes/claims_and_limitations.md) · [reproducibility](research/REPRODUCIBILITY.md)

### Key results so far (see the claims file before citing)
- **ASR pilot (10 utterances, 1 speaker):** Sarvam 2.0 % WER / 0.56 s · Whisper Tiny 4.5 % / 0.44 s · Whisper Base 6.5 % / 0.78 s. Semantic extraction (complete): 100 / 100 / 90 %.
- **Detector:** fine-tuned YOLOv8n, 15 classes, mAP50 0.481 on a COCO train2014 hold-out. This is optimistic because the val images likely overlap the pretraining data.
- **Desk trials:** the target-specific threshold plus smoothing removed most dropouts (22 → 0 lost events at 2 m). A face behind the object is the main failure.
- **Datasets:** COCO train2014, 604,906 boxes (person 30.6 %); RefCOCO, 133,784 unique expressions for 50,000 objects (person 50.2 %). Stage-1 documents report 267,568 expressions because both split schemes were counted ([correction](research/dataset/README.md)).

## 4. Limitations
- The ASR benchmark has 10 controlled single-speaker English utterances, with no noisy-environment test.
- Hardware tests were done by one user at a desk. There is no user study and no measured navigation success rate.
- "Left/right" is in image space; "reached" is a box-area heuristic. There is no metric distance.
- Objects outside the 15 classes are rejected; parser attributes (color/position) are not used yet.
- Speech recognition needs internet (Sarvam).

## 5. Setup & running

Python 3.13 (64-bit) on Windows; an NVIDIA GPU is optional (CPU works, slower).

```powershell
# One-time setup, from the repository root
python -m venv .venv
.\.venv\Scripts\python -m pip install torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m spacy download en_core_web_sm
copy .env.example .env      # then put your SARVAM_API_KEY in .env

# Verify
.\.venv\Scripts\python -c "import torch; print('CUDA:', torch.cuda.is_available())"
cd module1_audio; ..\.venv\Scripts\python -m pytest tests -q; cd ..
cd module2_vision; ..\.venv\Scripts\python -m pytest tests -q; cd ..

# Run (from module2_vision/)
cd module2_vision
..\.venv\Scripts\python main.py --target bottle --navigate                  # typed target, spoken guidance
..\.venv\Scripts\python main.py --target bottle --navigate --no-obstacles   # desk test: webcam faces you
..\.venv\Scripts\python assistant.py                                        # fully voice-driven session
```

Stop a session with **ENTER** in the terminal, **Q** in the video window, or Ctrl+C. It also stops automatically on arrival or after 60 s without seeing the target.
The camera must face the same direction as the user (e.g. a phone or chest camera) for "left/right" to be correct.

Data for training and the notebooks goes in `data/raw/` (see [notebooks/README](module2_vision/notebooks/README.md)); reproduction steps are in [research/REPRODUCIBILITY.md](research/REPRODUCIBILITY.md). Never commit `.env` or anything under `data/`.
