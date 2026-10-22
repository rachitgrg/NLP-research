# research/: source of truth for the paper

Everything the paper needs lives here: what was done (`LOG.md`), why (`decisions.md`), the data (`dataset/`), each experiment with its settings and outputs (`experiments/`), paper-ready figures and tables with the scripts that made them (`results/`, `figures/`), comparisons (`baselines/`), literature (`references/`), and writing notes (`paper_notes/`).

## Rules
1. **Never overwrite a result.** Add a new dated file or experiment folder; mark old ones as superseded.
2. **Every result records how it was made:** script or notebook, config, dataset version, date, environment.
3. After every significant event (new data or pipeline change, class list, training run, metrics, model change, result-affecting bug fix, scope decision, baseline), add a `LOG.md` entry, save the artifacts in the right folder, update `decisions.md` if needed, and update this index.
4. Before citing anything, check `paper_notes/claims_and_limitations.md`.

## Where to start
| Need | Go to |
|---|---|
| What happened, and when | `LOG.md` |
| Why something is the way it is | `decisions.md` |
| Which numbers are safe to cite | `paper_notes/claims_and_limitations.md` |
| Source of each figure and table | `results/MANIFEST.md` |
| Current verified system status | `PROJECT_AUDIT.md` (snapshot of 2026-10-20) |
| How to reproduce | `REPRODUCIBILITY.md` |

## Index

| Path | Contents |
|---|---|
| `README.md`, `LOG.md`, `decisions.md` | this index, dated changelog, design decisions |
| `PROJECT_AUDIT.md` | evidence-labelled audit of the whole system, repairs, hardware test and trials (2026-10-20) |
| `REPRODUCIBILITY.md` | environment and commands to reproduce the ASR benchmark, dataset processing and detector training |
| `dataset/README.md` | dataset versions, schema, statistics, class list, splits, **RefCOCO double-count correction** |
| `dataset/eda/` | notebook 03 EDA: 14 plots + `eda_summary.csv`, `expression_statistics.csv`, `category_distribution.csv` |
| `dataset/figures/` | notebook 01/02 inspection plots and verified examples |
| `dataset/MODULE_2_WORK_SUMMARY.md` | narrative of notebooks 01–03 |
| `experiments/2026-09-15_asr_pilot_2model/` | EXP-01, invalid (silent audio); kept for history |
| `experiments/2026-09-16_asr_benchmark/` | EXP-02 ASR + semantic eval; frozen `results_2026-09-16.csv`; Stage-1 audit |
| `experiments/2026-10-20_yolov8n_15class_finetune/` | EXP-03 training config, `results.csv`, curves, confusion matrices, val examples |
| `experiments/2026-10-20_manual_hardware_test/` | EXP-04 14-item functional test |
| `experiments/2026-10-20_desk_detection_trials/` | EXP-05 threshold/smoothing trials, fine-tuned vs stock |
| `experiments/2026-10-23_open_vocab_comparison/` | EXP-06 open-vocabulary detectors (R3): speed, vocabulary, desk accuracy (watch / keys / slippers); YOLOE-26s text prompt chosen |
| `experiments/2026-10-23_prompt_wording/` | EXP-07 prompt words for watch and keys: "watch" best; "bunch of keys" finds keys on a table (0 → 99.7 %); held in hand fails for both |
| `results/MANIFEST.md` | figure/table → script, inputs, caveats |
| `results/tables/` | tables 01–08 (Markdown) |
| `results/scripts/` | `generate_figures.py`, `generate_tables.py` (`--out` for check runs) |
| `results/STAGE_1_RESEARCH_FINDINGS.md` | Stage-1 master findings (2026-09-16) |
| `figures/` | fig_01–13 (300 dpi) |
| `baselines/README.md` | ASR and detector comparisons: settings, missing baselines |
| `references/` | `references.bib` (verify before use), reading notes |
| `paper_notes/` | `outline.md`, `claims_and_limitations.md`, `todo.md`, `PAPER_RESULTS_DISCUSSION.md` (draft), `FACULTY_DEMO_SUMMARY.md`, `IMPLEMENTATION_ROADMAP.md` |

## Old → new paths (restructure of 2026-10-21)
Documents written before 2026-10-21 use the old paths. They were left unchanged as historical records.

| Old | New |
|---|---|
| `module-1/` | `module1_audio/` |
| `module-3/` | `module2_vision/` |
| `module-3/audio_feedback/` | `module1_audio/audio_feedback/` |
| `module-2/train.py`, `test_model.py`, `dataset.yaml`, `results_summary.txt`, `job.sh`, `run_training.bat` | `module2_vision/training/` |
| `module-2/runs/` | `module2_vision/training/runs/` |
| `module-2/ipynb/0*.ipynb` | `module2_vision/notebooks/` |
| `module-2/datasets/coco2014`, `…/refcoco` | `data/raw/coco2014`, `data/raw/refcoco` |
| `module-2/outputs/*.csv` | `data/processed/` |
| `module-2/outputs/eda/`, `module-2/outputs/*.png` | `research/dataset/eda/`, `research/dataset/figures/` |
| `module-2/yolo_dataset/` | `data/yolo_dataset/` |
| `module-1/.env` | `.env` (project root) |
| `research/tables/`, `research/generate_*.py` | `research/results/tables/`, `research/results/scripts/` |
| `PROJECT_AUDIT.md`, `IMPLEMENTATION_ROADMAP.md`, `STAGE_1_RESEARCH_AUDIT.md` (root) | `research/`, `research/paper_notes/`, `research/experiments/2026-09-16_asr_benchmark/` |
