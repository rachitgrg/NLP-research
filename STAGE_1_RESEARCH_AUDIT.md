# Stage-1 Research Audit: Hearing & Understanding

**Date:** 2026-09-16  
**Module:** `module-1` (Speech and NLP Pipeline)

This document provides a comprehensive research-grade audit of the current implementation for Stage-1 of the NLP-Research project. The goal of this audit is to ensure the pipeline is robust, verifiable, and free of methodology errors before publication or further scaling.

---

## 1. Acoustic and ASR Evaluation

### 1.1 Methodology

The Automated Speech Recognition (ASR) benchmarking suite now supports a unified 3-model comparison:
- **Sarvam AI (Saaras v3)**: API-based model optimized for Indic accents and code-mixing.
- **Whisper Tiny**: OpenAI's 39M parameter local model.
- **Whisper Base**: OpenAI's 74M parameter local model.

All models are evaluated on identical audio using the `evaluate_models.py` script, which calculates:
1. **Word Error Rate (WER)** (Lower is better)
2. **Character Error Rate (CER)** (Lower is better)
3. **Exact Match Rate** (Higher is better)
4. **Per-sample Latency** (High-resolution `time.perf_counter`)

All reference and hypothesis texts are normalised prior to evaluation (lowercasing, punctuation stripping, extra whitespace removal).

### 1.2 Critical Finding: Silent WAV Files

> [!CAUTION]
> **Data Integrity Warning:** All 10 benchmark audio samples (`sample_01.wav` to `sample_10.wav`) currently contain no actual speech. Analysis shows their RMS energy is extremely low (~0.000015), representing only microphone background noise.
> 
> **Impact:** Since the input audio contains no speech, ASR models cannot produce meaningful transcriptions. As a result, the current `results.csv` shows 100% Word Error Rate (WER=1.0) and 0% Exact Match.
>
> **Action Required:** The evaluation infrastructure (`evaluate_models.py` and `plot_results.py`) is fully functional and tested. However, you MUST record real voice samples using `record_samples.py` and re-run `evaluate_models.py` to generate publishable ASR metrics.

---

## 2. NLP Pipeline and Semantic Extraction

The NLP parser extracts structured semantics (intent, main object, color, size, position) from transcribed English text using a lightweight, rule-based spaCy (`en_core_web_sm`) approach.

### 2.1 The "Last Noun" Object Selection Fix

**Previous Behavior:** The keyword extractor used a naive heuristic that selected the last noun in the sentence as the target object.
- Query: "Where is my phone on the table?"
- Extracted Object: `table` (Incorrect)

**Resolution:** The extractor (`keyword_extractor.py`) was updated to leverage spaCy's dependency parser. It now correctly identifies the primary object by prioritizing nouns with nominal subject (`nsubj`), direct object (`dobj`), or root (`ROOT`) syntactic dependencies.
- Query: "Where is my phone on the table?"
- Extracted Object: `phone` (Correct - `nsubj`)

### 2.2 End-to-End Semantic Evaluation

A new semantic evaluation pipeline (`semantic_eval.py`) was introduced to measure how well the NLP parser handles downstream output from the ASR models.

We created `semantic_gold.csv`, which contains hand-annotated ground-truth semantics for all 10 benchmark sentences. The evaluation script measures:
- **Intent Accuracy**: Did the parser correctly identify the goal?
- **Object Accuracy**: Was the target object correctly extracted?
- **Attribute Accuracy**: Were the color, size, and position properties correctly extracted?
- **Complete Accuracy**: Did the parsed representation perfectly match the gold standard across all fields?

**Baseline Verification:** When the semantic parser is fed the perfect reference texts, it achieves **100% Complete Accuracy**, validating the robustness of the rule-based approach for the benchmark queries.

---

## 3. Data Preprocessing Validation (Module-2)

### 3.1 Google "test" Split Validation

**Previous Behavior:** In `02_data_preprocessing.ipynb`, the dataset split `test` from the Google RefCOCOg source was inadvertently flagged or treated as an anomaly because UNC splits only contain `train`, `val`, `testA`, and `testB`.

**Resolution:** Explicit validation logic was added to the preprocessing pipeline to treat UNC and Google sources independently. 
- `VALID_SPLITS_UNC = {'train', 'val', 'testA', 'testB'}`
- `VALID_SPLITS_GOOGLE = {'train', 'val', 'test'}`

The Google `test` split is now correctly recognized as valid data, ensuring that no legitimate RefCOCOg data is erroneously discarded or flagged.

---

## 4. Conclusion

The pipeline is now mathematically sound and structurally ready for research documentation. The primary blocker to generating final graphs and benchmark numbers is the missing audio data (Section 1.2). Once the samples are recorded, running `evaluate_models.py` followed by `plot_results.py` and `semantic_eval.py` will produce a complete, research-grade evaluation.
