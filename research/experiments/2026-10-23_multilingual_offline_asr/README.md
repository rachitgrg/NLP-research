# EXP-08 · Offline multilingual speech recognition (roadmap R9)

**Status (2026-10-27): run 1 complete for the quiet condition: 120 Hindi + English clips, 3 speakers, every Whisper size (tiny → large-v3) offline plus the Sarvam cloud baseline.** Still to come: noisy-condition clips. Optional: AI4Bharat IndicConformer. Cite only run 1 and its caveats.

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
3. **Target set:** ~~Hindi plus every language with an available native speaker (aim ≥ 4)~~ **Hindi and English only (D-18, 2026-10-25)**, ≥ 2 speakers per language, quiet and noisy conditions. Sarvam baseline only within the free credit. Speakers are anonymous (S01, S02, …).
4. **Microphone (2026-10-25):** every clip is recorded on the laptop's built-in microphone array (Realtek), the microphone the assistant itself uses, so speakers are comparable. Bluetooth headsets are not used: their call-mode microphone is narrow-band, and on 2026-10-25 a recording froze while one was the default input (the recorder now times out instead of hanging, `evaluation/record_samples.py`).
5. **Recording:** `python evaluation/record_multilingual.py --language hi --speaker S01 [--condition noisy]`. Audio (5 s, 16 kHz mono) stays local (`*.wav` is gitignored); `multilingual/recordings.csv` is the manifest. Commit audio only with the speaker's consent.
6. **Run:** `python evaluation/multilingual_benchmark.py --backends whisper:small whisper:medium whisper:large-v3 whisper:large-v3+hint [sarvam --allow-cloud]`. It writes dated `evaluation/runs/multilingual_<date_time>.csv`, `…_summary.csv` and `….json` (settings, versions, commit) and never overwrites.

## Run 1: Hindi + English, 3 speakers (2026-10-27)
- **Data:** `run_2026-10-27/recordings_manifest_2026-10-27.csv`. 120 clips: speakers S01–S03, each says all 20 prompts in English and in Hindi, quiet room, laptop built-in microphone, 5 s, 16 kHz mono. Audio stays local (gitignored).
- **Run:** `cd module1_audio; OFFLINE_MODE=true ..\.venv\Scripts\python evaluation\multilingual_benchmark.py --backends <one backend>`. Each backend ran in its own process (see *Crash* below). Commit `475aa3c` plus an uncommitted manifest (the recordings). RTX 3050, `int8_float16`, beam 5, faster-whisper 1.2.1, Python 3.13.1.
- **Files:** `run_2026-10-27/multilingual_2026-10-27_<time>.csv` (per clip), `…_summary.csv`, `….json` (settings). Times: large-v3 `002346`, large-v3+hint `002602`, tiny `002730`, tiny+hint `002758`, base `002823`, base+hint `002856`.

| Backend | Object acc. EN (n=60) | Object acc. HI (n=60) | All (n=120) | LID acc. HI | Median latency | Max latency |
|---|---|---|---|---|---|---|
| `whisper:tiny` | 76.7 % | 33.3 % | 55.0 % | 80.0 % | 0.10 s | — |
| `whisper:tiny+hint` | 76.7 % | 33.3 % | 55.0 % | (given) | 0.08 s | — |
| `whisper:base` | 83.3 % | 55.0 % | 69.2 % | 85.0 % | 0.15 s | — |
| `whisper:base+hint` | 83.3 % | 55.0 % | 69.2 % | (given) | 0.11 s | — |
| `whisper:large-v3` | **91.7 %** | **90.0 %** | **90.8 %** | 60.0 % | 0.92 s | 2.22 s |
| `whisper:large-v3+hint` | 90.0 % | **90.0 %** | 90.0 % | (given) | **0.55 s** | 1.73 s |
| `whisper:small` | 88.3 % | 81.7 % | 85.0 % | 88.3 % | 0.27 s | 0.81 s |
| `whisper:small+hint` | 88.3 % | 81.7 % | 85.0 % | (given) | 0.17 s | 0.64 s |
| `whisper:medium` | 86.7 % | **90.0 %** | 88.3 % | 83.3 % | 0.56 s | 0.84 s |
| `whisper:medium+hint` | 86.7 % | **90.0 %** | 88.3 % | (given) | 0.35 s | 1.40 s |
| `sarvam` (cloud, Saaras v3) | 95.0 % | 96.7 % | 95.8 % | 100 % | 0.47 s | 0.92 s |

small/medium were downloaded on 2026-10-27 with user approval (`Systran/faster-whisper-small` snapshot `536b0662`, 464 MB; `-medium` snapshot `08e178d4`, 1.5 GB) and run offline. Sarvam ran on the free credit (120 calls, ~10 min of audio) at commit `1acbd66`. Its latency includes the network round trip from the laptop. Its JSON records `offline_mode: true`, which is misleading: that is the environment flag, and the Sarvam backend always uses the network. Files: small `004050`, small+hint `004131`, medium `004200`, medium+hint `004315`, sarvam `004427`.

Per speaker, large-v3 (correct out of 20): EN S01 19, S02 18, S03 18; HI S01 20, S02 18, S03 16. Large-v3 load time 7–8.5 s per model (once per process).

**Errors of large-v3 (11 of 120):** all are mishearings of the object word, not parser failures:
- *cup* (4): "Fine makeup", "cap", and in Hindi the object was dropped ("When will you find me?", "Look for it.").
- *bowl* (3): "bone", "ball", Hindi "Tori Dundo".
- *umbrella*, Hindi "chhata" (2): "Tata", "data". The Hindi word is translated as a name.
- *keys* (1): "Likey's"; *bag*, Hindi (1): "Hello, everyone." (a typical Whisper hallucination on a clip it can't decode).
- +hint adds one English error ("black button" for "black bottle", S03) and otherwise makes the same errors.

**What it shows:**
- Hindi needs at least `medium`: large-v3 and medium reach 90 %, small 81.7 %, base 55 %, tiny 33 %. English is easier for the small models (77–88 %), but large-v3 is still best.
- Giving the language doesn't change object accuracy here (identical for tiny and base; ±1 clip for large-v3), but it makes large-v3 ~40 % faster (median 0.92 → 0.55 s; max 2.22 → 1.73 s). Without the hint, large-v3 labels 40 % of the Hindi clips as another language (mostly code-mixed Hindi labelled English) and still translates most of them correctly.
- large-v3+hint meets the decision rule (≤ 2 s per 5 s clip, max 1.73 s). large-v3 without the hint has one clip at 2.22 s.
- **Offline vs cloud:** Sarvam is ahead by 5–6 points (95.8 % vs 90.0–90.8 %). On the same clips, Sarvam is right on every clip large-v3+hint gets right, plus 7 more (paired exact McNemar p = 0.016; large-v3 auto 6 vs 0, p = 0.031; medium 9 vs 0, p = 0.004; small 14 vs 1, p = 0.001). The two share their hardest clips: cup/bowl in S02/S03 English and S03 Hindi.
- **small and medium:** medium ties large-v3 in Hindi (90.0 %) and is 1.6× faster with the hint (0.35 s), but it is worse in English (86.7 %); small loses 8 points in Hindi. Without the hint, medium's English errors include mishearings such as "ballot" and "sleepers".
- **Decision (D-15, by the protocol's rule):** English: large-v3+hint 90.0 % is the best within ≤ 2 s (large-v3 auto, 91.7 %, has a 2.22 s clip). Hindi: large-v3 and medium tie at 90.0 %. So **large-v3 with the language given** for both. medium+hint (1.5 GB instead of 3.1 GB) is the fallback for smaller devices: 90 % in Hindi, 86.7 % in English.

**Caveats:** the McNemar p-values are uncorrected for six comparisons. With Bonferroni ×6, only medium (0.024) and small (0.006) stay below 0.05; large-v3 (0.19) and large-v3+hint (0.10) do not. Read the gap to Sarvam as consistent in direction (0 clips where Whisper large-v3 wins) but not yet significant after correction. 3 speakers (the developer and two others), one room, quiet only, one microphone, one take per prompt; 60 clips per language gives roughly ±8 percentage points (95 % interval) at 90 %. No significance test. Speakers chose their own Hindi wording. The gold object is checked after `normalize_target`, so synonyms such as *spectacles → glasses* count as correct.

**Crash (tooling):** the first attempt ran all six backends in one process. It exited with no error message during large-v3 (after tiny and base had been loaded and released) and wrote no files. Run alone, large-v3 completed (exit 0). The script already releases each model (`del transcriber`), so a native crash in CTranslate2 when models are swapped on the GPU is likely (not verified). Workaround: one backend per run. No result is affected, because results are only written when a run completes.

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
