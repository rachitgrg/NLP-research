# Dataset record

Produced by notebooks `module2_vision/notebooks/01_dataset_inspection.ipynb`, `02_data_preprocessing.ipynb` and `03_eda.ipynb` (Python 3.13.1 kernel, global Python; outputs dated 2026-09-15/16). Notebook narrative: `MODULE_2_WORK_SUMMARY.md`.

## ⚠ Correction (2026-10-21): RefCOCO expressions are counted twice
`refs(unc).p` and `refs(google).p` in `refcoco/refcoco/` are the **same RefCOCO annotations under two split schemes** (UNC: train/val/testA/testB; Google: train/val/test). They are not RefCOCO+ or RefCOCOg. Notebook 02 stacked both files and deduplicated on `(ann_id, expression, source)`, so every expression appears once per scheme. Checked on 2026-10-21 against `data/processed/refcoco_processed.csv`:

| | Reported in Stage-1 docs | Actual unique |
|---|---|---|
| Expressions | 267,568 (133,784 unc + 133,784 google) | **133,784** |
| Referred objects (`ann_id`) | 50,000 | 50,000 (same set in both schemes) |
| Tokens; spatial/color/size term counts | e.g. 961,454 tokens, "left" 63,060 | **≈ half** |
| Percentages and averages (person 50.17 %, mean 3.6 words) | — | unchanged |

Affected outputs (left unchanged as historical records): `research/results/tables/table_05..07`, `research/figures/fig_08..10` (absolute counts; rankings are unchanged), `eda/eda_summary.csv` (RefCOCO `total_rows`, `total_vocab_tokens`), and the "Final Expressions" lines in `STAGE_1_RESEARCH_FINDINGS.md`, `PAPER_RESULTS_DISCUSSION.md` and `FACULTY_DEMO_SUMMARY.md`. Any future grounding experiment must use **one** split scheme; mixing them leaks test expressions into training. See `research/LOG.md` (2026-10-21) and `decisions.md` D-09.

## Sources and versions

| Dataset | Files (in `data/raw/`) | Version / date |
|---|---|---|
| COCO 2014 | `coco2014/train2014/` (82,783 JPEGs, 12.6 GB); `coco2014/annotations/` instances, captions and person_keypoints for train2014 + val2014 (0.79 GB) | COCO 2014 release (annotation files dated 2017-09-02). **val2014 images are not downloaded.** |
| RefCOCO | `refcoco/refcoco/instances.json`, `refs(unc).p`, `refs(google).p` (0.17 GB) | refer-toolkit release (files dated 2016-02-04) |

Downloaded on 2026-09-14 (folder date). The download source URL was not recorded.

## Processed files (`data/processed/`, gitignored, regenerate with notebook 02)

`coco_processed.csv`: one row per COCO train2014 box, 604,906 rows (1 zero-height box removed from 604,907).

| Column | Meaning |
|---|---|
| image_id, image_path, width, height | COCO image |
| ann_id | COCO annotation id |
| bbox_x, bbox_y, bbox_width, bbox_height | COCO box in pixels (x, y = top-left) |
| category_id, category_name | COCO category (80 classes) |

`refcoco_processed.csv`: one row per expression per split scheme, 267,568 rows (see the correction above).

| Column | Meaning |
|---|---|
| image_id, image_path, ann_id | COCO image and the referred object |
| bbox_x, bbox_y, bbox_width, bbox_height | box of the referred object (from COCO) |
| category_id, category_name | category of the referred object |
| referring_expression | whitespace-normalised expression |
| split | UNC: train / val / testA / testB · Google: train / val / test |
| source | `unc` or `google` (split scheme) |

## Key statistics (`eda/eda_summary.csv`)
- COCO train2014: 82,081 images with at least one box, 604,906 boxes, 80 categories, 7.37 boxes per image on average (max 93). `person` = 185,315 boxes (30.6 %); the imbalance ratio largest/smallest class is 1,372.7.
- Box sizes: 44,614 boxes under 16 px; median relative area 1.2 %.
- RefCOCO: 19,994 images (24.4 % of COCO train2014 images), 50,000 referred objects, 78 categories. Expressions average 3.6 words (median 3, max 39). Person = 50.17 % of expressions.

Expression counts per split (`eda/expression_statistics.csv`; each scheme covers the same expressions):

| Scheme | train | val | testA | testB | test |
|---|---|---|---|---|---|
| unc | 113,373 | 10,268 | 5,349 | 4,794 | — |
| google | 107,002 | 13,414 | — | — | 13,368 |

## Class list used by the detector (15 classes, `module2_vision/training/dataset.yaml`)
0 person · 1 chair · 2 dining table · 3 couch · 4 bed · 5 bottle · 6 cup · 7 bowl · 8 laptop · 9 cell phone · 10 book · 11 tv · 12 backpack · 13 umbrella · 14 suitcase

The obstacle classes are 0–4; the rest are find targets (rationale in `decisions.md` D-03). Detector train/val split: see `research/experiments/2026-10-20_yolov8n_15class_finetune/`.

## Folder contents

| Path | What | Produced by |
|---|---|---|
| `eda/*.png`, `eda/*.csv` (14 plots, 3 tables) | COCO + RefCOCO EDA | notebook 03, 2026-09-16 |
| `figures/coco_*.png`, `figures/refcoco_verified_examples.png` | inspection plots and verified bbox/expression examples | notebooks 01 and 02, 2026-09-15/16 |
| `MODULE_2_WORK_SUMMARY.md` | step-by-step account of notebooks 01–03 | written 2026-09-15 |

These are snapshots. Re-running the notebooks writes to `module2_vision/notebooks/outputs/` (gitignored) and `data/processed/`. Add new snapshots here with a date; don't overwrite these.
