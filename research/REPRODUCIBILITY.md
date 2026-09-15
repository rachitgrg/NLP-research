# Reproducibility and Data Provenance

This document details the exact environment and steps required to reproduce the Stage-1 benchmarking and dataset processing.

## 1. Environment
- **Operating System:** Windows
- **Python Version:** 3.13.1
- **Key Dependencies:** `sarvamai`, `spacy`, `faster-whisper`, `jiwer`, `matplotlib`, `pytest`
- **Models:**
  - Sarvam AI Saaras v3 (Cloud API)
  - OpenAI Whisper Tiny (Local, `faster-whisper` implementation)
  - OpenAI Whisper Base (Local, `faster-whisper` implementation)
  - spaCy `en_core_web_sm` (Local NLP parsing)

## 2. Evaluation Data (Audio)
The benchmarking relies on 10 controlled audio utterances stored in `module-1/evaluation/audio/`.
- **Format:** `.wav`
- **Sampling Rate:** 16 kHz
- **Duration:** Exactly 5.0 seconds per sample
- **Source:** Single-speaker indoor recording

## 3. Data Provenance Classifications
- **A. Measured Experimentally:** ASR transcription outputs, Word Error Rate (WER), Character Error Rate (CER), Exact Match rate, and transcription latency.
- **B. Computed from Processed Datasets:** Semantic Evaluation accuracy (computed by running the NLP parser on the experimentally measured ASR outputs against the gold standard annotations), all COCO/RefCOCO statistics, and Linguistic EDA frequencies.
- **C. Qualitative Research Observations:** ASR error analysis (determining whether an error destroyed semantic meaning) and architectural conclusions.

## 4. Reproducing the ASR & Semantic Benchmarks

1. **Activate the Environment:**
   ```powershell
   cd d:\NLP-Research\module-1
   .\venv\Scripts\Activate.ps1
   ```

2. **(Optional) Re-record the benchmark samples:**
   If you wish to test new audio instead of the verified dataset:
   ```powershell
   python evaluation\record_samples.py
   ```

3. **Run the ASR Benchmark:**
   This command transcribes the audio using all 3 models, computes the metrics, and generates `evaluation/results.csv`.
   ```powershell
   python evaluation\evaluate_models.py
   ```

4. **Generate the Comparison Graph:**
   This reads `results.csv` and outputs `evaluation/comparison_graph_v2.png`.
   ```powershell
   python evaluation\plot_results.py
   ```

5. **Run the Semantic Evaluation:**
   This evaluates the semantic preservation of the ASR outputs against `evaluation/semantic_gold.csv`.
   ```powershell
   python evaluation\semantic_eval.py
   ```

6. **Run the Unit Test Suite:**
   There are exactly 77 automated tests covering the NLP parser, pipeline logic, and evaluation metrics.
   ```powershell
   pytest tests\ -v
   ```
   *Expected Output: 77 passed.*

## 5. Reproducing the Dataset Findings (Module-2)

The dataset processing and EDA were conducted via Jupyter notebooks in `module-2/ipynb/`.

1. **Dataset Inspection:** 
   Run `01_dataset_inspection.ipynb` to observe raw JSON structures and verify source counts (e.g., 604,907 original annotations).
2. **Data Preprocessing:** 
   Run `02_data_preprocessing.ipynb` to merge COCO and RefCOCO, validate bounding boxes, enforce explicit split checking for UNC/Google sources, and generate the deduplicated CSVs.
3. **Exploratory Data Analysis (EDA):** 
   Run `03_eda.ipynb` on the cleaned CSVs to generate the linguistic tokenization stats, unique word counts, spatial/color/size term frequencies, and the massive `person` category distributions.
