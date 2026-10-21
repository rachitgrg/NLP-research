# EXP-02 · ASR benchmark (3 models) + semantic evaluation

| | |
|---|---|
| Date | 2026-09-16 (`results.csv` file date; commit `6726126`) |
| Produced by | `module1_audio/evaluation/evaluate_models.py` → `results.csv`; `plot_results.py` → `comparison_graph_v2.png`; `semantic_eval.py` (prints to console only) |
| Paper figures/tables | `research/results/scripts/generate_{figures,tables}.py` → `research/figures/fig_02..07`, `research/results/tables/table_01..03` |
| Audio | 10 WAVs in `module1_audio/evaluation/audio/` (16 kHz, mono, 5.0 s each, single speaker, controlled indoor, English). Re-recorded on 2026-09-16 after the silent-file finding in `STAGE_1_RESEARCH_AUDIT.md` §1.2. |
| References | `module1_audio/evaluation/references.csv` (transcripts), `semantic_gold.csv` (intent / object / color / size / position, hand-annotated) |
| Models | Sarvam Saaras v3 (`model="saaras:v3"`, `mode="translate"`, `language_code="unknown"`, cloud API); Whisper Tiny and Whisper Base via faster-whisper (`device="cpu"`, `compute_type="int8"`, `task="translate"`, language auto-detect) |
| Metrics | WER and CER (jiwer, after lowercasing, punctuation removal and whitespace collapsing), exact match, latency (`time.perf_counter`, per file) |
| Environment | Windows, Python 3.13.1, `module-1/venv` (deleted 2026-10-21; package versions match the root `requirements.txt` pins) |

## Results (`results_2026-09-16.csv`, frozen copy of `module1_audio/evaluation/results.csv`)

| Model | WER | CER | Exact match | Mean latency | Median latency |
|---|---|---|---|---|---|
| Sarvam | 2.00 % | 1.36 % | 90 % | 0.560 s | 0.497 s |
| Whisper Tiny | 4.50 % | 3.24 % | 80 % | 0.441 s | 0.416 s |
| Whisper Base | 6.50 % | 3.69 % | 80 % | 0.779 s | 0.768 s |

Semantic evaluation (rule parser applied to each model's transcripts vs `semantic_gold.csv`):

| Input | Intent | Object | Attribute | Complete |
|---|---|---|---|---|
| Reference text | 100 % | 100 % | 100 % | 100 % |
| Sarvam | 100 % | 100 % | 100 % | 100 % |
| Whisper Tiny | 100 % | 100 % | 100 % | 100 % |
| Whisper Base | 90 % | 100 % | 100 % | 90 % |

The semantic numbers were originally only printed to the console. On 2026-10-21 they were re-run from the same `results.csv` with the same parser and matched exactly: `semantic_eval_rerun_2026-10-21.txt`.

## Conclusion
In this controlled 10-utterance pilot, Sarvam had the lowest WER and Whisper Tiny the lowest latency. Sarvam was chosen for the app (see `research/decisions.md` D-01).

## Caveats
- n = 10 utterances from one speaker, in English only, recorded in a quiet room. There is no significance testing. Present this as a pilot, not as a benchmark.
- Sarvam latency includes the network round trip; Whisper ran on CPU (int8).
- ⚠ Re-running `evaluate_models.py` overwrites `module1_audio/evaluation/results.csv`. The frozen copy in this folder is the version of record.
