# Notebooks — Dataset Inspection, Preprocessing & EDA

> Part of `module2_vision/`. These notebooks load, clean and explore the COCO 2014 and RefCOCO datasets.
> (Moved from `module-2/README.md` on 2026-10-21; paths updated.)

---

## What These Notebooks Do

They prepare two large computer vision datasets for downstream model training:

1. **COCO 2014**: a large image dataset with object annotations and captions.
2. **RefCOCO**: short referring expressions (natural-language phrases that describe a specific object in a COCO image, e.g. *"the man on the left"*).

The notebooks load the raw data, clean it, verify its integrity, and produce analysis outputs. `../training/train.py` builds the detector dataset from notebook 02's `coco_processed.csv`.

---

## Folder Structure

```
module2_vision/notebooks/
├── 01_dataset_inspection.ipynb    # Load and inspect the raw data
├── 02_data_preprocessing.ipynb    # Clean, validate, and export as CSV
├── 03_eda.ipynb                   # Exploratory data analysis and visualisation
└── outputs/                       # Figures written on re-run (gitignored)

data/                              # project root, gitignored (override with DATA_DIR)
├── raw/
│   ├── coco2014/
│   │   ├── train2014/             # 82,783 JPEG images
│   │   └── annotations/           # COCO JSON annotation files
│   └── refcoco/
│       └── refcoco/
│           ├── instances.json     # Object annotations for RefCOCO images
│           ├── refs(unc).p        # RefCOCO expressions, UNC split scheme (pickle)
│           └── refs(google).p     # The same RefCOCO expressions, Google split scheme (pickle)
└── processed/
    ├── coco_processed.csv         # Flattened COCO annotations (~52 MB), notebook 02
    └── refcoco_processed.csv      # Flattened RefCOCO expressions (~29 MB), notebook 02
```

Recorded results (EDA snapshot of 2026-09-16, statistics, schema): `research/dataset/`.

---

## Important Files

| File | What It Does |
|---|---|
| `01_dataset_inspection.ipynb` | Loads raw COCO and RefCOCO files. Verifies counts (82,783 images, 604,907 annotations, 50,000 RefCOCO objects). Checks 100 % cross-reference integrity. |
| `02_data_preprocessing.ipynb` | Flattens nested JSON to rows. Removes 1 invalid bounding box. Removes 16,852 duplicate expressions within each split scheme. Saves clean CSVs to `data/processed/`. |
| `03_eda.ipynb` | Exploratory analysis: category distribution plots, bounding box size analysis, expression length stats, word frequency charts, and a data quality audit. |
| `research/dataset/MODULE_2_WORK_SUMMARY.md` | Plain-language summary of everything the notebooks produced. |

---

## How to Run

Run the notebooks **in order** (01 → 02 → 03). Each notebook depends on the outputs of the previous one.

1. Put the datasets in `data/raw/` as shown above (see *Dataset Requirements*), or point the `DATA_DIR` environment variable at another folder that has the same layout.
2. Install the notebook dependencies (pinned in the root `requirements.txt`, "Training + notebooks" section): `pandas`, `numpy`, `matplotlib`, `pillow`, `ipykernel`.
3. Start Jupyter **with this folder as the working directory**, because the path cell resolves the project root from it:
   ```powershell
   cd module2_vision\notebooks
   jupyter notebook
   ```

Re-running overwrites `data/processed/*.csv` and `outputs/`. The recorded versions are in `research/dataset/`; save new results there as new dated snapshots.

---

## Dataset Requirements

> **The datasets are NOT included in this repository** (~15 GB). Download them separately.

| Dataset | Size | Source |
|---|---|---|
| COCO 2014 train images | ~13 GB | [cocodataset.org](https://cocodataset.org/#download) — "2014 Train images" |
| COCO 2014 annotations | ~241 MB | [cocodataset.org](https://cocodataset.org/#download) — "2014 Train/Val annotations" |
| RefCOCO | ~176 MB | [GitHub: lichengunc/refer](https://github.com/lichengunc/refer) |

---

## Expected Outputs

| File | Description |
|---|---|
| `data/processed/coco_processed.csv` | 604,906 rows, one per COCO object annotation |
| `data/processed/refcoco_processed.csv` | 267,568 rows, one per expression **per split scheme** (133,784 unique expressions) |
| `outputs/eda/*.png` | 13 analysis plots (category distributions, bounding box sizes, etc.) |
| `outputs/eda/*.csv` | 3 summary statistics tables |

---

## Key Findings (from the notebooks)

- **COCO 2014 Train**: 82,081 unique images, 604,906 annotations across 80 categories
- **RefCOCO**: 19,994 unique images, 50,000 objects, 133,784 unique referring expressions (the notebooks report 267,568 because both split schemes are kept; see `research/dataset/README.md`)
- **Category imbalance**: "person" makes up 30 % of COCO annotations and 50 % of RefCOCO expressions
- **Expression length**: Average 3.6 words (median 3.0); very short phrases
- **Data issues fixed**: 1 invalid bounding box removed, 16,852 duplicate expressions removed

---

## Limitations

- Only COCO 2014 **training images** are present. Validation images are not downloaded.
- `refs(unc).p` and `refs(google).p` are two split schemes of the **same** RefCOCO data (not RefCOCO+ or RefCOCOg). Use only one of them for any train/test experiment.
- The Google `test` split label is flagged by the data quality audit, but it is a valid label of the Google scheme.
