# CLAUDE.md

Assistive object finder for visually impaired users: voice request → camera detection → spoken direction guidance.
Research record (source of truth for the paper): `research/` (start at `research/README.md`). Verified system status: `research/PROJECT_AUDIT.md`. Planned work: `research/paper_notes/IMPLEMENTATION_ROADMAP.md`.

## Layout
- `module1_audio/`: all sound. Mic capture, Sarvam STT (translate to English), spaCy rule parser, ASR benchmark (`evaluation/`), speech output (`audio_feedback/`, settings in `audio_feedback/tts_config.py`). Entry: `main.py`.
- `module2_vision/`: all vision. Detection, matching, guidance, navigation, voice bridge (`pipeline/m3_pipeline.py`), trial tool (`tools/`). Entry points: `main.py` (flags `--target/--voice/--navigate`), `assistant.py` (voice-first).
  - `training/`: `train.py` (15-class YOLOv8n fine-tune), `test_model.py`, `runs/` (gitignored; `runs/train/weights/best.pt` is the default model).
  - `notebooks/`: 01 inspection, 02 preprocessing, 03 EDA (run with `notebooks/` as the working directory).
  - `weights/`: pretrained `yolov8n.pt` (gitignored).
- `data/` (gitignored, override with the `DATA_DIR` env var): `raw/coco2014`, `raw/refcoco` (read-only downloads), `processed/` (notebook 02 CSVs), `yolo_dataset/` (hard links + labels from `train.py`).
- `research/`: paper source of truth. See *Research record* below.
- `_to_delete/`: items awaiting the user's review. Don't use them or delete them without asking.

## Environment
- Use the root `.venv` (Python 3.13, CUDA torch, voice + vision deps): `.\.venv\Scripts\python`. Training and notebooks ran under the global Python 3.13 (pandas, scikit-learn, ipykernel are not yet in `.venv`).
- Secrets: the root `.env` (`SARVAM_API_KEY`). Never print, commit, or copy it. `.env.example` holds names only.

## How to run
- Voice only: `cd module1_audio; ..\.venv\Scripts\python main.py`
- Assistant: `cd module2_vision; ..\.venv\Scripts\python main.py --target bottle --navigate` (desk test: add `--no-obstacles`) or `..\.venv\Scripts\python assistant.py`
- ASR benchmark: `cd module1_audio; ..\.venv\Scripts\python evaluation\evaluate_models.py` (calls the paid Sarvam API; ask first). Writes `evaluation/runs/results_<date_time>.csv`; `evaluation/results.csv` is the 2026-09-16 run of record and is never written. Plot or score a run with `plot_results.py --results <csv>` / `semantic_eval.py --results <csv>`.
- Detection trials: `cd module2_vision; ..\.venv\Scripts\python tools\detection_trials.py --scene "<name>"` (appends to `tools/results/detection_trials_<date>.csv` with the settings used; `detection_trials.csv` is the frozen 2026-10-20 record)
- Open-vocabulary (R3): benchmark `tools\openvocab_benchmark.py`; desk trials `tools\openvocab_trials.py --target watch --scene "<name>"` (all models on the same recorded frames; nothing saved to disk; `--prompts a b a+b` compares prompt words, EXP-07). Weights and text encoders live in `module2_vision/weights/`; `OpenVocabDetector` downloads text encoders there, not to Ultralytics' global folder.
- Training: `cd module2_vision\training; python train.py` (ask first; overwrites `runs/train/`)
- Paper tables/figures: `.\.venv\Scripts\python research\results\scripts\generate_tables.py --out <scratch>` (without `--out` it overwrites committed outputs; ask first)

## Tests
- `cd module1_audio; ..\.venv\Scripts\python -m pytest tests -q` (96 tests on 2026-10-21)
- `cd module2_vision; ..\.venv\Scripts\python -m pytest tests -q` (194 tests on 2026-10-23)
- Tests mock the camera, YOLO, and TTS. Hardware behaviour needs a manual run.

## Conventions
- Run scripts from inside their module folder; each module puts its own root on `sys.path`.
- Both modules have a `pipeline` package. `module2_vision` loads Module 1's `hearing_pipeline.py` by file path (`pipeline/m3_pipeline.py`); never `import pipeline.hearing_pipeline` from `module2_vision`.
- `module2_vision` imports TTS (`audio_feedback`) from `module1_audio` via `config.add_module1_to_path()`, which **appends** to `sys.path` so the vision packages keep priority. Call it before importing `audio_feedback`.
- Detection dict contract: `{"class_name","class_id","confidence","bbox":[x1,y1,x2,y2]}`. Don't change it without updating all consumers.
- Supported targets come from the loaded model (`detector.class_names`), not from the COCO-80 list. Choose the detector for a target with `detection/detector_factory.prepare_target_detector()` (closed-set for its classes; otherwise the open-vocabulary model alongside it, D-13). Don't build detectors for a target by hand in entry points.
- Vision tunables live in `module2_vision/config/config.py`, TTS tunables in `module1_audio/audio_feedback/tts_config.py`. All can be overridden with env vars.
- Paths: derive them from `__file__` (never hard-code `D:\...`); data paths go through `DATA_DIR`.
- Naming: experiment folders `research/experiments/YYYY-MM-DD_<short_name>/`; dated snapshots `<name>_YYYY-MM-DD.<ext>`; decisions `D-NN`; experiments `EXP-NN`.
- Match the existing style: header comment block per file, type hints, small focused classes.

## Research record (do this without being asked)
- Treat `research/` as the single source of truth for the paper.
- After any significant event, update `research/` **in the same session**. Significant events include:
  - a new dataset or a change to the dataset pipeline
  - a new or changed class list or schema
  - a new experiment or training run
  - new metrics or results
  - a model or architecture change
  - a bug fix that affects results
  - a scope or decision change
  - a new baseline comparison
- For each event:
  1. Add a dated entry at the top of `research/LOG.md` (what happened, why, links to files and commits).
  2. Save the artifacts (metrics, plots, configs, frozen CSVs) in the right subfolder. Each experiment gets `research/experiments/<date>_<name>/README.md` with script, config, dataset version, environment, date, metrics, conclusion and caveats.
  3. Update `research/decisions.md` if a decision changed: mark the old entry superseded and add a new one.
  4. Update `research/README.md` (index), `research/results/MANIFEST.md` (if a figure or table changed) and `research/paper_notes/claims_and_limitations.md` (if a claim changed).
- Never overwrite old results. Version them (new dated file or folder) so earlier numbers stay traceable.
- Every result file must record how it was produced: script or notebook, config, dataset version, date.
- If the user finishes something big and `research/` wasn't updated, remind them.

## Safety rules
- Never commit `.env` or raw datasets (`data/` is gitignored; keep it that way).
- Never delete or overwrite `data/`, `module2_vision/training/runs/`, `*.pt`, notebooks, or `research/` outputs.
- Ask before installing packages, downloading weights or datasets, or (re)training.
- Don't claim distance or navigation accuracy that hasn't been measured. Check `research/paper_notes/claims_and_limitations.md` before stating results.
