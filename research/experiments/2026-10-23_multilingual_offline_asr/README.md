# EXP-08 · Offline multilingual speech recognition (roadmap R9)

**Status (2026-10-23): protocol and tooling ready; only an English smoke test has run.** The multilingual recordings and the large-model downloads are still to come. Don't cite any multilingual number from this folder yet.

| | |
|---|---|
| Question | Which on-device recogniser should replace Sarvam (D-01), per language? How far below the cloud baseline is it? |
| Headline metric | **Object accuracy**: audio → recogniser (speech → English) → spaCy parser → the app's `normalize_target` → equals the gold object. Alternatives are allowed, e.g. `keys|key`. This is what decides whether the assistant searches for the right thing. |
| Secondary | Language-ID accuracy, latency per request (median), model load time, model size on disk |
| Candidates | Whisper `small`, `medium`, `large-v3` (faster-whisper, `task="translate"`), each auto-detect and `+hint` (language given). Sarvam Saaras v3 as the cloud reference (paid; `--allow-cloud`). Optional: AI4Bharat IndicConformer-600M → IndicTrans2-200M (needs install approval; plan Phase 1.6). Excluded: `large-v3-turbo` (not trained for translation); SeamlessM4T / MMS (CC-BY-NC). |
| Decision rule | The highest object accuracy per language, subject to ≤ 2 s ASR latency for a 5 s clip on the RTX 3050. A per-language route is allowed. The result becomes D-15 final. |
| Scripts | `module1_audio/evaluation/record_multilingual.py` (recording + manifest), `module1_audio/evaluation/multilingual_benchmark.py` (benchmark), `module1_audio/evaluation/multilingual/prompts.csv` (20 requests) |
| Environment | root `.venv`, Python 3.13.1, faster-whisper 1.2.1 (CTranslate2), RTX 3050 6 GB, `OFFLINE_MODE=true` |

## Protocol
1. **Prompts:** 20 find-requests (`prompts.csv`). They cover all 15 closed-set targets/obstacles plus open-vocabulary objects (watch, keys, slippers, glasses, wallet, medicine, charger), one with a colour attribute, and two where the object isn't the last noun ("phone on the table", "cup near the laptop").
2. **Speakers say each request naturally in their own language.** English loanwords such as "phone" are fine, and code-mixing is expected. The Hindi column is only a suggestion.
3. **Target set:** Hindi plus every language with an available native speaker (aim ≥ 4), ≥ 2 speakers per language, quiet and noisy conditions. Speakers are anonymous (S01, S02, …).
4. **Recording:** `python evaluation/record_multilingual.py --language hi --speaker S01 [--condition noisy]`. Audio (5 s, 16 kHz mono) stays local (`*.wav` is gitignored); `multilingual/recordings.csv` is the manifest. Commit audio only with the speaker's consent.
5. **Run:** `python evaluation/multilingual_benchmark.py --backends whisper:small whisper:medium whisper:large-v3 whisper:large-v3+hint [sarvam --allow-cloud]`. It writes dated `evaluation/runs/multilingual_<date_time>.csv`, `…_summary.csv` and `….json` (settings, versions, commit) and never overwrites.

## Smoke test (2026-10-23 04:57, commit `1d7bb32`; the script itself was committed unchanged in `154d4af`)
The 10 EXP-02 English clips (`en_smoke_manifest.csv`, gold objects from `semantic_gold.csv`) with the two cached models, on the GPU (`int8_float16`, beam 5):

| Backend | Object accuracy | Language-ID accuracy | Median latency |
|---|---|---|---|
| `whisper:tiny` | 10/10 | 9/10 | 0.096 s |
| `whisper:base` | 10/10 | 9/10 | 0.129 s |
| `whisper:tiny+hint` | 10/10 | 10/10 | 0.078 s |

Files: `multilingual_2026-10-23_045753.csv` (per clip), `…_summary.csv`, `….json`.

**What it shows:** the tooling works end to end, offline (`OFFLINE_MODE=true`, models from the local cache), and GPU inference is ~4–5× faster than the CPU numbers of EXP-02 (0.44 s). Auto language-ID labelled one Indian-English clip as Hindi; giving the language fixes that, which is why `ASR_LANGUAGE` exists.
**What it does not show:** anything about Hindi or other languages. It is one speaker, the same 10 clips as EXP-02, and too small to compare models.

## Findings while building the tooling (affect any re-run of EXP-02)
- faster-whisper 1.2.1 calls `av.open(..., metadata_errors=...)`, which PyAV 19.0.1 (installed in `.venv`) no longer accepts. Every Whisper transcription failed with `TypeError` in the project environment. EXP-02 (2026-09-16) ran in the old `module-1/venv` and is unaffected. Fixed: audio is read with soundfile (`speech/whisper_stt.py` `load_audio`).
- On Windows, CTranslate2 didn't find cuBLAS (`cublas64_12.dll`), so Whisper could only run on CPU. Fixed by registering the CUDA torch wheel's `torch/lib` (`register_cuda_libraries`).
