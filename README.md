# Multimodal Audio-Visual Grounding Framework for Real-Time Assistive Navigation
*(Stage-1 Implementation)*

An end-to-end NLP and computer vision research project designed to build a real-time assistive navigation system for blind and low-vision users.

---

## 1. Project Overview & Stage-1 Scope

This repository currently implements **Stage 1** of the navigation framework. 
It establishes the two foundational pillars required before visual grounding can occur:

1. **Hearing & Understanding (Linguistic Pipeline)** — Capturing a spoken voice command, transcribing it to English, and extracting the user's intent, the target object, and its attributes using a rule-based NLP parser.
2. **Dataset Preparation (Vision Foundation)** — Processing, cleaning, and validating the massive COCO 2014 and RefCOCO datasets to create a high-quality referring-expression dataset for future multimodal model training.

> [!WARNING]
> **Future Work:** Full visual grounding, bounding-box prediction, multimodal spatial reasoning, and real-time camera processing are scheduled for **Stage 2** and are *not* implemented in this repository.

---

## 2. Architecture

**1. Linguistic Pipeline (Implemented):**  
`Microphone → ASR (Sarvam/Whisper) → English Transcription → spaCy NLP Parser → Intent/Object/Attributes → [Future Visual Grounding]`

**2. Dataset Foundation (Implemented):**  
`COCO + RefCOCO → Validation → Preprocessing → Exploratory Data Analysis (EDA) → [Future Grounding Dataset]`

*(See [System Architecture Diagram](research/figures/fig_01_stage1_architecture.png) and [Dataset Pipeline Diagram](research/figures/fig_13_dataset_pipeline.png))*

---

## 3. Completed Modules

- **`module-1/`**: Voice pipeline containing microphone capture, a multi-model ASR framework, and the `spaCy` NLP parser. Includes a robust 77-case automated test suite.
- **`module-2/`**: Jupyter notebooks dedicated to COCO and RefCOCO dataset inspection, bounding-box validation, split verification, and Linguistic EDA.
- **`research/`**: The comprehensive academic research output of Stage 1, including formal findings, a faculty demonstration summary, and publication-ready graphs and tables.

---

## 4. Research Documents & Findings

For a complete breakdown of the research methodology, dataset EDA, and analytical findings, please read the academic documentation:

- 📄 **[Master Stage-1 Research Findings](research/STAGE_1_RESEARCH_FINDINGS.md)** *(Single source of truth)*
- 📄 **[Faculty Demonstration Summary](research/FACULTY_DEMO_SUMMARY.md)** *(5-minute presentation overview)*
- 📄 **[Paper Results & Discussion Draft](research/PAPER_RESULTS_DISCUSSION.md)** *(Publication draft)*
- 📄 **[Reproducibility & Data Provenance](research/REPRODUCIBILITY.md)**

### Key Benchmarks (10-Utterance Controlled Benchmark)

#### ASR Benchmark Summary
Evaluated on Word Error Rate (WER) and Mean Latency:
- **Sarvam AI (Cloud):** 2.00% WER | 0.56s Latency *(Best Accuracy)*
- **Whisper Tiny (Local):** 4.50% WER | 0.44s Latency *(Best Latency)*
- **Whisper Base (Local):** 6.50% WER | 0.77s Latency

*(See [Accuracy vs Latency Tradeoff](research/figures/fig_06_accuracy_latency_tradeoff.png) and [ASR WER Comparison](research/figures/fig_02_asr_wer.png))*

#### Semantic Evaluation Summary
Measures preservation of information needed by the downstream parser across the transcription outputs:
- **Reference / Sarvam / Whisper Tiny:** 100% Complete Semantic Accuracy
- **Whisper Base:** 90% Complete Semantic Accuracy

*(See [Semantic Accuracy Comparison](research/figures/fig_07_semantic_accuracy.png))*

#### COCO/RefCOCO Dataset Findings
- **Cleaned Data:** 267,568 final expressions referring to 50,000 unique targets.
- **Integrity:** 0 referential-integrity mismatches found after validation.
- **Bias Alert:** Massive category imbalance detected. `person` accounts for ~30.6% of COCO annotations and ~50.2% of RefCOCO expressions, necessitating mitigation during Stage-2 training.

*(See [RefCOCO Spatial Language](research/figures/fig_08_refcoco_spatial_terms.png) and [RefCOCO Category Imbalance](research/figures/fig_12_refcoco_category_distribution.png))*

---

## 5. Limitations

- The ASR and semantic benchmark is limited to 10 controlled, single-speaker, English utterances. No real-world noisy environment or statistically massive audio benchmark has been conducted.
- The NLP parser relies on rigid syntactic rules and struggles with complex multi-object conjunctions.
- Downstream visual grounding is not yet implemented.

---

## 6. Reproducibility

To reproduce the Stage-1 benchmarking and tests:

```powershell
# 1. Activate Environment
cd module-1
.\venv\Scripts\Activate.ps1

# 2. Run the ASR Benchmark (Requires SARVAM_API_KEY in .env)
python evaluation\evaluate_models.py

# 3. Generate the Comparison Graphs
python evaluation\plot_results.py

# 4. Run the Semantic Evaluation
python evaluation\semantic_eval.py

# 5. Run the Unit Test Suite (77 passing tests)
pytest tests\ -v
```

See [REPRODUCIBILITY.md](research/REPRODUCIBILITY.md) for full data provenance details and dataset generation instructions.
