# EXP-01 · ASR pilot, 2 models (Whisper Tiny vs Sarvam) — INVALID, do not cite

| | |
|---|---|
| Date | 2026-09-15 (graph file date; commit `cdf59d2`, 2026-09-16) |
| Produced by | `module-1/evaluation/evaluate.py` (now `_to_delete/module-1/evaluation/evaluate.py`; also at tag `pre-restructure-2026-10-21`) |
| Input | `module-1/evaluation/audio/sample_01..10.wav` as recorded on 2026-09-15, `references.csv` |
| Models | Whisper Tiny (faster-whisper), Sarvam Saaras v3 |
| Output | `comparison_graph.png` (this folder). The per-sample `results.csv` it wrote was overwritten by EXP-02. |

## Result
Both models scored **0.00 % average accuracy**.

## Why it is invalid
The 2026-09-16 audit (`../2026-09-16_asr_benchmark/STAGE_1_RESEARCH_AUDIT.md`, §1.2) found that all 10 WAV files at that time contained no speech (RMS ≈ 0.000015, background noise only). The 0 % reflects silent input, not model quality.

## Conclusion
Kept only as a record of the pipeline bug that led to re-recording the audio. The valid benchmark is EXP-02.
