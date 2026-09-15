# Results and Discussion

## 1. Experimental Setup

To evaluate the linguistic front-end of the proposed audio-visual grounding framework, we established a controlled 10-utterance benchmark. The benchmark sentences were selected to represent typical assistive navigation queries, encompassing varied spatial, color, and size attributes. Audio was recorded in a controlled indoor environment from a single speaker at a 16 kHz sampling rate, with each sample strictly truncated to 5.0 seconds. 

We compared three Automatic Speech Recognition (ASR) approaches: a cloud-based model (Sarvam AI Saaras v3) and two local models (OpenAI Whisper Tiny and Whisper Base). The models were evaluated on Word Error Rate (WER), Character Error Rate (CER), Exact Match rate, and transcription latency using high-resolution timers.

Following transcription, the English outputs were parsed using a lightweight, rule-based Natural Language Processing (NLP) pipeline built on the `spaCy` dependency parser. Semantic accuracy was measured by comparing the extracted intent, primary object, and visual attributes against a hand-annotated gold standard.

## 2. ASR Results

On the controlled 10-utterance benchmark, the cloud-based Sarvam AI model demonstrated the highest transcription accuracy, achieving a WER of 2.00% and an Exact Match rate of 90.00%. However, this accuracy traded off against latency, with Sarvam recording a mean latency of 0.5602s.

The local Whisper Tiny (39M parameters) model exhibited the lowest mean latency (0.4413s), making it the most responsive candidate for real-time applications. Its transcription accuracy was marginally lower, with a WER of 4.50%. Whisper Base (74M parameters) performed poorly in this benchmark relative to its size, exhibiting both the highest WER (6.50%) and the highest mean latency (0.7786s).

## 3. Semantic Evaluation and Error Analysis

A qualitative error analysis of the individual ASR outputs reveals a crucial distinction between raw transcription accuracy (WER) and downstream semantic preservation. A transcription can contain errors that negatively impact WER without degrading the extraction of critical grounding entities.

For instance, in Sample 04 ("Is there a laptop in front of me?"), both Whisper models transcribed the utterance as "They are a laptop in front of me." While this yields a non-zero WER due to syntactic hallucination, the NLP parser successfully extracted the target object ("laptop") and the spatial attribute ("in front of"). Similarly, in Sample 08 ("Where is the blue book?"), grammatical substitutions such as "Where is my blue book?" (Sarvam/Tiny) or "Here is my blue book" (Whisper Base) did not prevent the parser from extracting the correct intent and attributes.

This semantic resilience is formally reflected in our Semantic Evaluation. While the Reference texts, Sarvam, and Whisper Tiny all achieved 100% Complete Semantic Accuracy on this benchmark, Whisper Base achieved 90% Complete Semantic Accuracy due to downstream parser failure on its specific transcription errors. 

## 4. Dataset Analysis

The visual grounding foundation was established by analyzing the COCO 2014 and RefCOCO datasets. Following validation and deduplication, the RefCOCO dataset yielded 267,568 unique referring expressions referencing 50,000 unique annotation targets across 19,994 images. Rigorous integrity checking revealed zero referential mismatches between the referring expressions and the underlying COCO bounding box annotations.

Exploratory Data Analysis (EDA) of the RefCOCO linguistic corpus highlighted its richness in spatial and descriptive language, which is essential for training robust grounding models. We observed 226,670 total occurrences of spatial terms (e.g., "left", "right", "front"), as well as high frequencies of color and size descriptors.

However, a significant category imbalance was identified. The `person` category constitutes approximately 30.64% of valid COCO annotations and an overwhelming 50.17% of all RefCOCO expressions. 

## 5. Discussion

The findings from Stage 1 establish a strong foundation for the real-time assistive navigation framework. RQ1 asked which ASR approach provides the optimal trade-off; our results suggest that while Sarvam offers superior transcription accuracy, Whisper Tiny's low latency and high semantic resilience make it a highly competitive local alternative for real-time edge computing. 

Addressing RQ2, the rule-based NLP pipeline proved highly effective at extracting intent and visual attributes, provided the syntactic structure of the ASR output remained relatively intact. Regarding RQ3 and RQ4, the dataset analysis confirms that RefCOCO contains the rich spatial language required for downstream grounding, but explicitly warns that the massive `person` category imbalance must be mitigated during multimodal training to prevent algorithmic bias in the navigation system.

## 6. Limitations

It is critical to acknowledge that the ASR and semantic accuracy results were obtained on a limited, 10-utterance English benchmark recorded by a single speaker in a controlled environment. These results lack large-scale statistical significance and do not reflect performance in noisy, real-world environments. Furthermore, downstream visual grounding (bounding-box prediction and real-time camera integration) has not yet been implemented, and the system has not been subjected to blind-user evaluation.

## 7. Stage-1 Conclusion

Stage 1 successfully implements the linguistic understanding front-end and prepares the necessary dataset foundations for multimodal visual grounding. The evaluation demonstrates that lightweight local models like Whisper Tiny can achieve sufficiently low latency and high semantic accuracy to support real-time interaction. With the data pipeline validated and the linguistic module complete, the framework is well-positioned to advance to Stage 2: multimodal fusion and spatial navigation.
