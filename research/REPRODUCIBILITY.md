# Reproducibility and Data Provenance

This document gives the environment and the steps to reproduce the Stage-1 benchmarks, the dataset processing and the detector training.

> Updated 2026-10-21 for the new repository layout (old → new paths: `README.md` in this folder). The original Stage-1 version (2026-09-16) is in git history at tag `pre-restructure-2026-10-21`.
> Since 2026-10-22 the ASR benchmark and the detection trials write new dated files and never touch earlier results. Training, the notebooks and the figure scripts (without `--out`) still overwrite their outputs. The versions of record are the frozen copies in `research/experiments/`; record new runs as new dated experiments.

## 1. Environment
- **Operating System:** Windows
- **Python Version:** 3.13.1
- **Environment:** root `.venv`, set up from `requirements.txt` (see the root `README.md`). Training and notebooks 01–03 ran under the global Python 3.13 (see `requirements.txt`, "Training + notebooks").
- **Models:**
  - Sarvam AI Saaras v3 (Cloud API; needs `SARVAM_API_KEY` in the root `.env`)
  - OpenAI Whisper Tiny (Local, `faster-whisper` implementation)
  - OpenAI Whisper Base (Local, `faster-whisper` implementation)
  - spaCy `en_core_web_sm` (Local NLP parsing)
  - Ultralytics YOLOv8n (`module2_vision/weights/yolov8n.pt`) and the fine-tuned `module2_vision/training/runs/train/weights/best.pt`

## 2. Evaluation Data (Audio)
The benchmarking relies on 10 controlled audio utterances stored in `module1_audio/evaluation/audio/`.
- **Format:** `.wav`
- **Sampling Rate:** 16 kHz
- **Duration:** Exactly 5.0 seconds per sample
- **Source:** Single-speaker indoor recording

## 3. Data Provenance Classifications
- **A. Measured Experimentally:** ASR transcription outputs, Word Error Rate (WER), Character Error Rate (CER), Exact Match rate, and transcription latency.
- **B. Computed from Processed Datasets:** Semantic Evaluation accuracy (computed by running the NLP parser on the experimentally measured ASR outputs against the gold standard annotations), all COCO/RefCOCO statistics, and Linguistic EDA frequencies.
- **C. Qualitative Research Observations:** ASR error analysis (determining whether an error destroyed semantic meaning) and architectural conclusions.

## 4. Reproducing the ASR & Semantic Benchmarks

Run from `module1_audio/` with the root environment:
```powershell
cd module1_audio
python evaluation\record_samples.py      # (optional) re-record the 10 samples
python evaluation\evaluate_models.py     # 3 models -> evaluation\runs\results_<date_time>.csv (new file per run)
python evaluation\plot_results.py --results evaluation\runs\results_<date_time>.csv   # -> same name .png
python evaluation\semantic_eval.py --results evaluation\runs\results_<date_time>.csv  # prints accuracy vs semantic_gold.csv
# without --results both scripts read evaluation\results.csv (the 2026-09-16 run of record)
python -m pytest tests -q                # unit tests
```
Paper figures and tables: from the project root run `python research\results\scripts\generate_tables.py` and `generate_figures.py`. Add `--out <folder>` to write somewhere other than `research/`.

## 5. Reproducing the Dataset Findings

Notebooks are in `module2_vision/notebooks/` and must be started with that folder as the working directory. Raw data is expected in `data/raw/` (override with the `DATA_DIR` environment variable).

1. **`01_dataset_inspection.ipynb`:** raw JSON structures and source counts (e.g. 604,907 original annotations).
2. **`02_data_preprocessing.ipynb`:** flattens COCO and RefCOCO, validates boxes and splits, deduplicates per split scheme, and writes `data/processed/{coco,refcoco}_processed.csv`. ⚠ Both RefCOCO split schemes are kept, so each expression appears twice (`research/dataset/README.md`).
3. **`03_eda.ipynb`:** EDA plots and tables, written to `module2_vision/notebooks/outputs/eda/`. The 2026-09-16 snapshot is in `research/dataset/eda/`.

## 6. Reproducing the Detector Training

From `module2_vision/training/` (needs `pandas`, `scikit-learn`, `tqdm`, `ultralytics` and a CUDA GPU):
```powershell
python train.py                      # builds data\yolo_dataset (hard links) + dataset.yaml, trains 30 epochs
python test_model.py --image <img>   # quick check of runs\train\weights\best.pt
```
The config used for the reported run is in `research/experiments/2026-10-20_yolov8n_15class_finetune/` (`args.yaml`, `dataset.yaml`).
