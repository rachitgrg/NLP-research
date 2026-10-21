# Multimodal Audio-Visual Grounding Framework for Real-Time Assistive Navigation
## Stage-1 Research Findings

**Author/Investigator:** Senior Research Engineer  
**Date:** September 2026

---

### 1. Research Title
Multimodal Audio-Visual Grounding Framework for Real-Time Assistive Navigation (Stage-1 Implementation)

### 2. Abstract / Executive Summary
This document summarizes the completed Stage-1 findings of the proposed assistive navigation framework. The broader motivation of this project is to create an assistive navigation system that allows blind or low-vision users to issue real-time voice commands (e.g., "Where is my blue bag?") and receive spatial grounding from a wearable camera. 

This Stage-1 research specifically covers the linguistic and dataset foundations required before visual grounding can be implemented. It implements and benchmarks a spoken-command understanding pipeline (ASR + NLP parser) and prepares the COCO/RefCOCO datasets for downstream multimodal training. We compare three ASR models (Sarvam AI, Whisper Tiny, Whisper Base) on a controlled benchmark to evaluate the trade-off between latency, transcription accuracy, and semantic preservation.

### 3. Research Problem
Real-time assistive navigation systems must rapidly and accurately parse spoken user intent in noisy or low-compute environments, while remaining resilient to ASR transcription errors that do not affect the core semantic object or attributes being sought. Furthermore, training these systems requires massive, high-quality visual-grounding datasets free of referential ambiguities and structural corruption.

### 4. Research Motivation
Assistive navigation holds the potential to significantly improve independence for low-vision users. However, existing frameworks often suffer from high latency in the cloud or poor transcription accuracy locally, breaking the real-time feedback loop. We are motivated to establish a robust linguistic pipeline and a clean, validated visual grounding dataset to ensure subsequent computer vision models have the highest possible quality of inputs.

### 5. Research Objectives
- Implement and benchmark an audio-to-semantics pipeline.
- Evaluate the exact impact of ASR errors on downstream semantic parsing.
- Extract, validate, and perform Exploratory Data Analysis (EDA) on the COCO and RefCOCO datasets to establish a clean foundation for future visual grounding.

### 6. Research Questions
**RQ1:** Which speech-recognition approach provides the best trade-off between transcription accuracy, latency, and preservation of object-oriented semantic information for a real-time assistive audio-visual grounding framework?  
**RQ2:** How effectively can a lightweight NLP pipeline extract intent, target object, and visual attributes from transcribed commands?  
**RQ3:** What data characteristics and quality issues arise when preparing COCO and RefCOCO for downstream referring-expression-based visual grounding?  
**RQ4:** To what extent does Stage 1 establish the linguistic and dataset foundations required for later audio-visual grounding?  

---

### 7. System Architecture
The Stage-1 architecture is divided into two distinct components:

**1. The Linguistic Pipeline:**
`Microphone → ASR (Sarvam/Whisper) → English Transcription → spaCy NLP Parser → Intent/Object/Attributes → [Future Visual Grounding]`

**2. The Dataset Foundation:**
`COCO + RefCOCO → Validation → Preprocessing → EDA → [Future Grounding Dataset]`

### 8. Stage-1 Implementation
The actual implemented modules and their responsibilities are:
- **Speech Module:** Interfaces with Sarvam AI's Saaras v3 API and local Whisper models to convert multilingual audio to English text.
- **NLP Module:** Uses a rule-based `spaCy` dependency parser to extract one of four intents (`find_object`, `describe_object`, `identify_object`, `general_question`), identify the target object (e.g., "phone"), and extract spatial, size, and color attributes.
- **Evaluation Module:** Automates the benchmarking of ASR performance and semantic accuracy.
- **Data Preprocessing (Module-2):** Jupyter notebooks that load, merge, clean, and analyze the COCO and RefCOCO dataset JSON/pickle files.

### 9. ASR Experimental Setup
This evaluation is a controlled/pilot benchmark, NOT a statistically broad ASR benchmark.
- **Models:** Sarvam AI Saaras v3 (Cloud), Whisper Tiny 39M (Local), Whisper Base 74M (Local)
- **Audio Format:** WAV, 16 kHz, single-channel
- **Evaluation Data:** 10 carefully controlled utterances
- **Recording Conditions:** Single speaker, controlled indoor environment, exactly 5.0 seconds per recording.
- **Evaluation Metrics:** Word Error Rate (WER), Character Error Rate (CER), Exact Match (Boolean), High-resolution Python `time.perf_counter` latency.

---

### 10. ASR RESULTS
The aggregate results measured on the 10-utterance benchmark are:

| Model | WER | CER | Exact Match | Mean Latency | Median Latency |
|---|---|---|---|---|---|
| **Sarvam** | 2.00% | 1.36% | 90.00% | 0.5602 s | 0.4967 s |
| **Whisper Tiny** | 4.50% | 3.24% | 80.00% | 0.4413 s | 0.4159 s |
| **Whisper Base** | 6.50% | 3.69% | 80.00% | 0.7786 s | 0.7677 s |

### 11. INDIVIDUAL ASR ERROR ANALYSIS
A qualitative analysis of transcription errors highlights the distinction between WER and semantic loss. A transcription can have a non-zero WER while still preserving the object and attributes required by the downstream NLP parser. Note that these errors were semantically preserved for this controlled benchmark, but we do not claim they are harmless in every possible real-world application.

**Sample 04 Analysis:**
- **Reference:** “Is there a laptop in front of me?”
- **Sarvam:** “Is there a laptop in front of me?” (Exact match)
- **Whisper Tiny / Base:** “They are a laptop in front of me.”
*Significance:* Despite substituting "Is there" with "They are", the target object ("laptop") and spatial position ("in front of") are perfectly preserved.

**Sample 08 Analysis:**
- **Reference:** “Where is the blue book?”
- **Sarvam / Tiny:** “Where is my blue book?”
- **Whisper Base:** “Here is my blue book.”
*Significance:* The substitution of "the" for "my", or "Where" for "Here" changes the syntactic structure but leaves the core query ("blue book") intact for parsing.

---

### 12. SEMANTIC EVALUATION
This evaluation measures the preservation of information needed by the downstream parser across the transcription outputs. *Note: The 100% values reported below are obtained specifically on this 10-utterance controlled benchmark and should not be interpreted as universal semantic accuracy.*

| Model | Intent | Object | Attribute | Complete |
|---|---|---|---|---|
| **Reference** | 100% | 100% | 100% | 100% |
| **Sarvam** | 100% | 100% | 100% | 100% |
| **Whisper Tiny** | 100% | 100% | 100% | 100% |
| **Whisper Base** | 90% | 100% | 100% | 90% |

### 13. NLP PARSER
The rule-based NLP pipeline uses `spaCy` (`en_core_web_sm`) to parse English text.
- **Intents Supported:** `find_object`, `describe_object`, `identify_object`, `general_question`.
- **Extraction:** Identifies the primary noun using dependency parsing (`nsubj`, `dobj`), effectively ignoring distractor nouns in locative prepositional phrases. Extracts colors, sizes, and specific spatial positions.
- **Limitations:** The parser currently struggles with multiple distinct target objects connected by conjunctions (e.g., "Find my keys and my wallet"), or highly ambiguous locative phrasing if standard dependency trees fail. 

---

### 14. COCO DATASET FINDINGS
The following statistics were verified during Stage-1 validation of the COCO 2014 dataset context:
- 82,783 images in the source context
- 604,907 original annotations
- 604,906 valid annotations after removing one invalid bounding box
- 80 categories
- 82,081 unique image IDs

### 15. RefCOCO DATASET FINDINGS
The RefCOCO dataset underwent deduplication and integrity validation:
- UNC references: 50,000 | Google references: 50,000
- 284,420 expressions before deduplication
- 16,852 duplicate expressions removed
- 267,568 final expressions
- 19,994 unique images
- 50,000 unique annotation targets
- 78 categories represented
- **0 referential-integrity mismatches**

*Note:* Google RefCOCO uses a “test” split while UNC uses “testA/testB”. This was validated as a legitimate split difference, not dataset corruption.

### 16. RefCOCO LINGUISTIC EDA
- **Tokens:** 961,454
- **Unique words:** 10,116
- **Average expression length:** 3.59 words (Median: 3 words)
- **Total spatial-term occurrences:** 226,670

**Top Spatial Terms:** left (63,060), right (62,326), front (15,814), bottom (13,704), middle (12,622)  
**Top Color Terms:** white (14,510), black (10,604), blue (10,574), red (9,894), green (5,072)  
**Top Size Terms:** big (1,980), little (1,172), small (910), tall (524), large (460)  

### 17. CATEGORY IMBALANCE
Both datasets exhibit significant categorical bias toward humans:
- **COCO:** The `person` category accounts for ≈ 30.64% of valid annotations (185,315 / 604,906).
- **RefCOCO:** The `person` category accounts for ≈ 50.17% of all expressions (134,252 / 267,568).
*Impact:* This massive imbalance must be mitigated or explicitly accounted for during downstream visual grounding training to prevent the model from overwhelmingly defaulting to human detection.

---

### 18. RESEARCH FINDINGS
**Finding 1:** Sarvam had the strongest transcription accuracy in the controlled benchmark.  
**Finding 2:** Whisper Tiny had the lowest latency.  
**Finding 3:** Whisper Base was slower than both Sarvam and Whisper Tiny and had the highest WER in this benchmark.  
**Finding 4:** Some ASR errors did not prevent semantic extraction because the important object/attribute information was preserved.  
**Finding 5:** RefCOCO contains rich spatial and descriptive language suitable for future referring-expression visual grounding.  
**Finding 6:** Both COCO and RefCOCO exhibit substantial person-category imbalance that should be considered in future experiments.  
**Finding 7:** The dataset integrity checks found no referential mismatches after preprocessing.  

### 19. RESEARCH CONTRIBUTIONS
**Implemented (Stage 1):**
- Speech-command understanding pipeline
- ASR comparison framework & Semantic evaluation
- COCO/RefCOCO preparation and dataset integrity validation
- Linguistic EDA
- Reproducible evaluation scripts and test suites

**Future Work (Stage 2+):**
- Visual grounding and bounding-box prediction
- Multimodal fusion and spatial reasoning
- Real-time camera experiments and navigation path planning
- Blind-user evaluation

### 20. LIMITATIONS
- Evaluation is restricted to a 10-utterance, single-speaker, English-only benchmark in a controlled environment.
- No large-scale statistical significance analysis was performed on the ASR metrics.
- No real-world noisy environment evaluation has been conducted.
- No blind-user study has been conducted.
- Downstream visual grounding is not yet implemented.

### 21. FUTURE WORK
Future stages will focus on leveraging the cleaned RefCOCO dataset to train multimodal fusion models capable of mapping the parsed semantic constraints directly to bounding boxes within real-time camera feeds, ultimately supporting physical navigation.

### 22. REPRODUCIBILITY
To reproduce the Stage-1 benchmarking and tests:
```bash
# 1. Activate Environment
.\module-1\venv\Scripts\Activate.ps1

# 2. Re-record Samples (Optional)
python module-1\evaluation\record_samples.py

# 3. Evaluate Models (ASR)
python module-1\evaluation\evaluate_models.py

# 4. Generate Graphs
python module-1\evaluation\plot_results.py

# 5. Run Semantic Evaluation
python module-1\evaluation\semantic_eval.py

# 6. Run Unit Tests (77 passing tests)
pytest module-1\tests\ -v
```
