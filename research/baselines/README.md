# Baselines and comparisons

Each comparison records its settings so it can be rerun under identical conditions. Results live in `../experiments/`; this file explains what was compared, and how.

## B-1 · Speech recognition: Sarvam vs Whisper Tiny vs Whisper Base
- **Experiment:** `../experiments/2026-09-16_asr_benchmark/`
- **System under test:** Sarvam Saaras v3, cloud API, `mode="translate"`, `language_code="unknown"`.
- **Baselines:** OpenAI Whisper Tiny (39 M) and Base (74 M) via faster-whisper 1.2.1, `device="cpu"`, `compute_type="int8"`, `task="translate"`, language auto-detect.
- **Shared settings:** the same 10 WAVs (16 kHz mono, 5 s), the same normalisation (lowercase, strip punctuation, collapse spaces), jiwer WER/CER, latency measured with `time.perf_counter` around one call per file.
- **Not controlled:** network latency for Sarvam. Model warm-up is not excluded.
- **Script:** `module1_audio/evaluation/evaluate_models.py`

## B-2 · Detector: fine-tuned 15-class YOLOv8n vs stock 80-class YOLOv8n
- **Experiments:** `../experiments/2026-10-20_desk_detection_trials/` (live webcam, two face scenes for stock) and `../experiments/2026-10-20_yolov8n_15class_finetune/` (fine-tuned only).
- **Desk-trial settings:** target `bottle`; base threshold 0.40; target threshold 0.25; `TargetTracker` 2-of-5 with 0.5 s hold; 640 px; CUDA; 15 s per scene. Stock run with `YOLO_MODEL_PATH=weights/yolov8n.pt` from `module2_vision/`.
- **Missing (roadmap R2):** both models on the same 15 classes on clean COCO **val2017** images (mAP50, mAP50-95 per class, FPS). Needs a val2017 download (approval required).

## Not compared yet
- Other assistants or apps (e.g. Seeing AI, Be My Eyes, Lookout): no comparison has been run. If one is added, record the app version, device, prompts and scenes here.
- Open-vocabulary detectors (YOLO-World, YOLOE, OWLv2, Grounding DINO): planned in roadmap R3.
