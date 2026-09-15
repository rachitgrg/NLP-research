# Faculty Demonstration Summary
**Multimodal Audio-Visual Grounding Framework for Real-Time Assistive Navigation**

## 1. Project Title
Multimodal Audio-Visual Grounding Framework for Real-Time Assistive Navigation (Stage-1)

## 2. Problem
Real-time assistive navigation systems must accurately parse spoken user intent in noisy environments. They must be resilient to ASR (speech-to-text) errors that do not affect core semantic meaning, and they require massive, high-quality, verified visual-grounding datasets to train downstream computer vision models.

## 3. Motivation
Assistive navigation can significantly improve independence for low-vision users. However, existing frameworks suffer from high cloud latency or poor local transcription accuracy, breaking the real-time feedback loop. Establishing a robust linguistic pipeline and a clean, validated visual grounding dataset is the necessary foundation for future real-world deployment.

## 4. What Has Actually Been Implemented
- **Speech-Command Pipeline:** Captures audio and transcribes it using Sarvam AI (cloud), Whisper Tiny, or Whisper Base.
- **Semantic Parser:** Uses a rule-based NLP approach (`spaCy`) to extract intent, object, and spatial/color/size attributes.
- **Evaluation Infrastructure:** Fully reproducible benchmarking for ASR (WER, Latency) and Semantic Accuracy.
- **Dataset Preparation:** Automated validation, cleaning, deduplication, and integrity checking of the COCO and RefCOCO datasets.

## 5. Stage-1 Architecture
**1. Linguistic Pipeline:**  
Microphone → ASR (Sarvam/Whisper) → English Transcription → spaCy NLP Parser → Intent/Object/Attributes → *(Future Visual Grounding)*

**2. Dataset Foundation:**  
COCO + RefCOCO → Validation → Preprocessing → EDA → *(Future Grounding Dataset)*

## 6. ASR Comparison Table (10-Utterance Benchmark)
| Model | WER | CER | Exact Match | Mean Latency | Median Latency |
|---|---|---|---|---|---|
| **Sarvam** | 2.00% | 1.36% | 90.00% | 0.5602 s | 0.4967 s |
| **Whisper Tiny** | 4.50% | 3.24% | 80.00% | 0.4413 s | 0.4159 s |
| **Whisper Base** | 6.50% | 3.69% | 80.00% | 0.7786 s | 0.7677 s |

## 7. Semantic Evaluation Table
| Model | Intent | Object | Attribute | Complete |
|---|---|---|---|---|
| **Reference** | 100% | 100% | 100% | 100% |
| **Sarvam** | 100% | 100% | 100% | 100% |
| **Whisper Tiny** | 100% | 100% | 100% | 100% |
| **Whisper Base** | 90% | 100% | 100% | 90% |

## 8. Dataset Findings
- **COCO:** 82,783 images, 604,906 valid annotations (80 categories).
- **RefCOCO:** 267,568 deduplicated expressions referencing 50,000 unique targets across 19,994 images.
- **Integrity:** 0 referential-integrity mismatches found.
- **Bias:** Massive `person` category imbalance (30% of COCO annotations, 50% of RefCOCO expressions).

## 9. Top 5 Research Findings
1. **Sarvam Accuracy:** Sarvam provided the strongest transcription accuracy (2% WER).
2. **Tiny Latency:** Whisper Tiny provided the lowest latency (0.44s mean).
3. **Semantic Resilience:** Specific ASR errors did not prevent successful semantic extraction (e.g., "They are a laptop" vs "Is there a laptop" both successfully yield "laptop").
4. **Spatial Richness:** RefCOCO contains massive amounts of spatial referential language (over 226,000 spatial term occurrences).
5. **Dataset Bias:** The extreme `person` imbalance in RefCOCO must be explicitly handled during Stage-2 training to prevent model bias.

## 10. Key Graphs to Show
*(Refer to `research/figures/` for high-resolution PNGs)*
- **Figure 1:** Stage-1 System Architecture
- **Figure 6:** Transcription Accuracy vs. Mean Latency
- **Figure 7:** Semantic Preservation Accuracy
- **Figure 12:** RefCOCO Category Distribution (Highlighting Imbalance)

## 11. Limitations
- Benchmark is restricted to 10 controlled, single-speaker, English utterances.
- No large-scale statistical significance analysis or noisy-environment evaluation.
- Downstream visual grounding is not yet implemented (Stage 2).

## 12. Future Work
- Visual grounding and bounding-box prediction.
- Multimodal fusion mapping semantic constraints to real-time camera feeds.
- Navigation path planning and real-world blind-user evaluation.

## 13. Reproduction Commands
```bash
# From repository root
.\module-1\venv\Scripts\Activate.ps1
python module-1\evaluation\evaluate_models.py
python module-1\evaluation\semantic_eval.py
pytest module-1\tests\ -v
```
