# Module 2 — Work Summary (Notebooks 01, 02, 03)

This document provides a comprehensive summary of all the work completed in `module-2/ipynb/`. It is based strictly on the code and outputs present in the notebooks.

## 2-Minute Presentation Version
We have successfully ingested, cleaned, and explored the COCO 2014 and RefCOCO datasets. 
- **Notebook 01** inspected the raw data formats, verifying that COCO contains 82.7k images and 604k object annotations, while RefCOCO contains 50k referring expressions for both UNC and Google splits. We verified the cross-referencing between COCO bounding boxes and RefCOCO expressions, achieving a 100% match. 
- **Notebook 02** standardized this data into two clean CSV files (`coco_processed.csv` and `refcoco_processed.csv`). We removed 1 invalid COCO bounding box and deduplicated 16.8k exact RefCOCO expressions, preserving all valid annotations. 
- **Notebook 03** performed deep exploratory data analysis (EDA). We analyzed bounding box sizes, category distributions (finding severe class imbalance towards "person"), referring expression lengths (averaging 3.6 words), and linguistic patterns (heavy use of spatial and color terms). All EDA plots and statistics have been saved for downstream use. There is no unexecuted code or unimplemented plans; the data pipeline is complete and ready for model training.

---

## 12-Step Summary of Executed Work

1. **Environment Setup & Path Validation:** Standardized file paths for datasets and outputs across all notebooks. Verified read-only access to raw datasets and write access to `module-2/outputs`.
2. **COCO 2014 Inspection:** Loaded `instances_train2014.json`. Found 82,783 images, 604,907 annotations, and 80 categories.
3. **RefCOCO Inspection:** Loaded RefCOCO `instances.json` (19,994 images, 196,771 annotations) and referring expressions `refs(unc).p` and `refs(google).p` (50,000 reference entries and 142,210 expressions each).
4. **Cross-Reference Validation:** Verified that 100% of RefCOCO annotations correctly map to valid COCO images and bounding boxes.
5. **COCO Flattening:** Converted the nested COCO JSON into a flat DataFrame (`coco_df`) with one row per annotation, containing bounding box coordinates and category labels.
6. **RefCOCO Flattening & Enrichment:** Expanded RefCOCO expressions into a flat DataFrame (`refcoco_df`) and enriched it with COCO image dimensions and bounding boxes.
7. **Expression Cleaning:** Normalised whitespace in referring expressions and removed empty strings (0 removed).
8. **Bounding Box Validation:** Checked bounding boxes for positive dimensions and within-image boundaries. Removed 1 invalid COCO bounding box (zero height).
9. **Deduplication:** Removed 16,852 exact duplicate referring expressions in RefCOCO (where `ann_id`, `referring_expression`, and `source` were identical). 
10. **Data Serialization:** Saved the cleaned DataFrames as `coco_processed.csv` (52.17 MB) and `refcoco_processed.csv` (28.92 MB).
11. **Comprehensive EDA:** Analyzed image dimensions, annotation densities, bounding box sizes (flagging 7% as tiny <16px), category distributions (severe imbalance, 30% of annotations are "person"), and referring expression lengths (mean 3.6 words).
12. **Quality Audit & Output Generation:** Ran a systematic 8-category data quality audit across both DataFrames. Generated and saved 13 EDA plots and 3 statistical CSVs to `outputs/eda/`.

---

## Detailed Breakdown by Notebook

### Notebook 01: `01_dataset_inspection.ipynb`
- **What was actually executed/completed:** 
  - Validated dataset directory structure and loaded raw COCO and RefCOCO files.
  - Extracted key statistics: 82,783 COCO images, 604,907 COCO annotations, 50,000 RefCOCO UNC references, 50,000 RefCOCO Google references.
  - Verified 5 examples visually, drawing bounding boxes and displaying the referring expressions.
  - Ran a large-scale referential integrity check: 100% of RefCOCO references correctly matched a COCO image and annotation.
- **Code written but not executed:** None. All cells have execution outputs.
- **Planned but not implemented:** None. All planned inspection tasks were fully implemented.

### Notebook 02: `02_data_preprocessing.ipynb`
- **What was actually executed/completed:**
  - Built efficient lookup dictionaries for fast merging.
  - Flattened COCO annotations into `coco_df` (604,907 rows).
  - Flattened and enriched RefCOCO expressions into `refcoco_df` (284,420 initial rows).
  - Cleaned text expressions (stripped whitespace, collapsed multiple spaces).
  - Validated bounding boxes: Removed 1 COCO row with zero height.
  - Checked referential integrity: 0 mismatches found.
  - Deduplication: Removed 16,852 exact duplicates in RefCOCO.
  - Saved outputs: `coco_processed.csv` (604,906 rows) and `refcoco_processed.csv` (267,568 rows).
  - Generated a preprocessing summary table confirming all row counts.
- **Code written but not executed:** None. All cells have execution outputs.
- **Planned but not implemented:** None. All planned preprocessing tasks were fully implemented.

### Notebook 03: `03_eda.ipynb`
- **What was actually executed/completed:**
  - Loaded preprocessed CSVs and verified their integrity (zero nulls in critical columns).
  - **COCO EDA:** Plotted image dimensions, annotations per image (mean 7.4), category distribution (severe imbalance: `person` has 185k annotations vs `hair drier` with 135, ratio 1372.7x), and bounding box areas (7.38% tiny boxes <16x16px).
  - **RefCOCO EDA:** Plotted split distribution, category distribution (`person` makes up 50.17% of expressions), and expression lengths (mean 18.2 chars, 3.6 words).
  - **Linguistic Analysis:** Analyzed a vocabulary of 10,116 unique words. Identified top domain-specific terms (spatial: 'left', 'right'; color: 'white', 'black'; size: 'big', 'little'; person: 'guy', 'man').
  - **Data Quality Audit:** Ran 56 quality checks. Found 1 minor issue: invalid split label `13368` (which actually indicates rows belonging to the 'test' split from Google, flagged because it checked against only train/val/testA/testB).
  - Saved 13 plots (PNG) and 3 summary tables (CSV) to `outputs/eda/`.
- **Code written but not executed:** None. All cells have execution outputs.
- **Planned but not implemented:** None. All planned exploratory analyses were fully implemented.

---

## Key Statistics & Findings
*(Sourced exclusively from verified notebook outputs)*

* **COCO 2014 Train Data:** 82,081 unique images, 604,906 object annotations, 80 categories.
* **RefCOCO Data:** 19,994 unique images (24.36% of COCO), 50,000 unique annotation targets (8.27% of COCO), 267,568 referring expressions.
* **Category Imbalance (COCO):** Severe. 'person' (185,315) vs 'hair drier' (135).
* **Category Focus (RefCOCO):** Extremely heavily skewed toward 'person' (50.17% of all expressions).
* **Expression Lengths:** Very short. Average 3.6 words (median 3.0 words). The shortest expressions are single words (e.g., 'standing', 'woman', 'middle'); the longest are up to 39 words.
* **Linguistic Patterns:** Heavy reliance on positional terms ('left' appears 63,060 times; 'right' 62,326 times).

## Data Quality / Limitations
- **Output Files:** All expected preprocessing CSVs and EDA plots/CSVs were successfully generated and saved.
- **Quality Issues Found:** 
  - 1 invalid COCO bounding box removed.
  - 16,852 duplicate RefCOCO expressions removed.
  - RefCOCO contains a `test` split from the Google source which was flagged as an invalid label during a quality check (since the check only expected `val`, `testA`, `testB`, `train`).
