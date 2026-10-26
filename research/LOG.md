# Research log

Newest first. Each entry: what happened, why, links. Entries before 2026-10-21 were reconstructed from git history and file dates. Old paths (`module-1/`, `module-2/`, `module-3/`) are mapped to new ones in `README.md`.

---

## 2026-10-27 — EXP-08 run 1 completed: Whisper small/medium and the Sarvam baseline; D-15 model size decided
- **What:** with user approval, downloaded Whisper `small` (464 MB) and `medium` (1.5 GB) and ran them offline, with and without the language hint, on the same 120 clips. Ran the Sarvam cloud baseline on the free credit (120 calls).
- **Results (object accuracy EN / HI, median latency):** small 88.3 % / 81.7 %, 0.27 s (hint 0.17 s); medium 86.7 % / 90.0 %, 0.56 s (hint 0.35 s); Sarvam 95.0 % / 96.7 %, 0.47 s (cloud, includes network). Paired against Sarvam, large-v3+hint: 7 clips Sarvam-only, 0 Whisper-only (exact McNemar p = 0.016, 0.10 after Bonferroni ×6).
- **Decision:** D-15 model size decided: large-v3 + language hint (best within 2 s in English; ties with medium in Hindi). medium+hint is the smaller-device fallback. Claims C20 updated and C21 (gap to cloud) added.
- **Caveat:** the Sarvam run's JSON says `offline_mode: true`, which is only the environment flag; that backend always uses the network.
- **Pending:** noisy-condition recordings (user).
- **Links:** `experiments/2026-10-23_multilingual_offline_asr/` (run 1, `run_2026-10-27/`), `decisions.md` D-15, `paper_notes/claims_and_limitations.md` C20–C21.

## 2026-10-27 — EXP-08 run 1: offline Hindi + English speech recognition (3 speakers)
- **What:** the user recorded 120 clips: speakers S01–S03, the 20 prompts in English and in Hindi, quiet room, laptop microphone (`module1_audio/evaluation/multilingual/recordings.csv`). Ran `evaluation/multilingual_benchmark.py` offline (`OFFLINE_MODE=true`, RTX 3050) with Whisper tiny, base and large-v3, each with auto language detection and with the language given (`+hint`).
- **Results (object accuracy EN / HI, median latency):** tiny 76.7 % / 33.3 %, 0.10 s; base 83.3 % / 55.0 %, 0.15 s; large-v3 91.7 % / 90.0 %, 0.92 s (max 2.22 s); large-v3+hint 90.0 % / 90.0 %, 0.55 s (max 1.73 s). The hint doesn't change accuracy, but it makes large-v3 ~40 % faster. All 11 large-v3 errors are misheard object words (cup, bowl, Hindi "chhata" → "Tata"/"data"), not parser failures.
- **Decision:** D-15 provisionally large-v3 + language hint (already the default model size). Not final until `small`/`medium` and Sarvam have been compared. Claim C20 added (indicative).
- **Tooling issue:** running all six backends in one process crashed silently during large-v3 (no files written). Run alone it completes, so each backend now runs in its own process. Likely a CTranslate2 model-swap crash (not verified). No results affected.
- **Pending:** Whisper `small`/`medium` (download, needs approval), Sarvam baseline (free credit, needs approval), noisy-condition clips.
- **Links:** `experiments/2026-10-23_multilingual_offline_asr/` (run 1 and `run_2026-10-27/`), `decisions.md` D-15, `paper_notes/claims_and_limitations.md` C20.

## 2026-10-25 — Scope for the paper: Hindi and English; Sarvam within free credit (D-18)
- **Why:** the user has made the paper the priority over finishing the product. The only languages the user and available speakers can record are Hindi and English, and the project has no budget for paid API calls.
- **What:** EXP-08 is limited to Hindi and English (≥ 2 speakers each); other languages become a stated limitation (claims file, limitation 10). The Sarvam baseline runs only within the account's free credit. No code changed.
- **Environment:** downloaded Whisper large-v3 (`Systran/faster-whisper-large-v3`, snapshot `edaa852e`, 3,091 MB, HF cache) with user approval; `tools/offline_check.py` now reports "Ready to run offline". Offline check (`OFFLINE_MODE=true`, RTX 3050, int8_float16): load 11.5 s; EXP-02 clips 01 and 08 transcribed correctly in 1.55 s (first call) and 0.94 s. A check only, not a result.
- **Recorder fix:** an EXP-08 recording froze in `sd.wait()` with Bluetooth earbuds as the default input; clips 1–7 were saved, clip 8 wasn't. `evaluation/record_samples.py` now times out after duration + 3 s and asks for a retry (2 tests; 123 voice tests pass). EXP-08 clips are recorded on the laptop's built-in microphone only. No results affected.
- **Links:** `decisions.md` D-18, `experiments/2026-10-23_multilingual_offline_asr/README.md`, `paper_notes/claims_and_limitations.md` C20.

## 2026-10-23 — R9 started: fully offline, multilingual assistant (branch `feature/offline-multilingual`)
- **Why:** new project direction (user, 2026-10-23). Offline, on-device operation is the USP compared with cloud assistants such as Gemini. The paper uses the laptop prototype; a phone/glasses product comes later without a rewrite now. The system must stay multilingual. Work happens on a branch so `main` stays demo-ready.
- **What (code, commits `1d7bb32`, `154d4af`, `8bccb95`, `b1969dc`):**
  - `OFFLINE_MODE` (default on): sets the Ultralytics / Hugging Face offline flags before the libraries load, and makes the open-vocabulary loader fail fast on missing files. `tools/offline_check.py` is the pre-flight (D-16).
  - On-device speech recognition with faster-whisper (`speech/whisper_stt.py`, `transcriber_factory.py`, `asr_config.py`) replaces Sarvam as the default; Sarvam is opt-in (D-15, supersedes D-01). The voice pipeline is now loaded once per process.
  - EXP-08 tooling: prompt sheet, recorder, multilingual benchmark measuring object accuracy.
  - Spoken replies in the user's language through `guidance/phrases.py` and voice selection by language (D-17).
  - A per-stage latency log for sessions (`navigation/session_log.py`, EXP-09).
- **Bugs found (affect re-running EXP-02, not its frozen results):**
  1. faster-whisper 1.2.1 + PyAV 19 in `.venv`: every Whisper transcription failed (`metadata_errors` `TypeError`). Audio is now read with soundfile.
  2. CTranslate2 on Windows didn't find cuBLAS, so Whisper ran on CPU only. `torch/lib` is now registered.
- **Results:**
  - EXP-08 English smoke test (10 EXP-02 clips, GPU): tiny/base 10/10 objects, median 0.08–0.13 s; auto language-ID 9/10, 10/10 with the language given.
  - EXP-10 (partial, CPU only): `best.pt` 22.6 ms at 640 px, YOLOE-26s 73.6 ms (29.2 ms at 320 px, with lower confidences); Whisper tiny/base 0.20/0.37 s.
  - No multilingual result yet.
- **Tests:** module1 96 → 121, module2 194 → 214, all pass.
- **Pending (need the user):**
  - Download Whisper `small`/`medium`/`large-v3`.
  - Record the multilingual test set.
  - Approve the Sarvam baseline calls.
  - Native-speaker review of the Hindi phrases, and a Windows Hindi voice.
  - EXP-09: run with the network off.
- **Links:** `experiments/2026-10-23_multilingual_offline_asr/`, `experiments/2026-10-23_edge_feasibility/`, `decisions.md` D-15–D-17, `paper_notes/IMPLEMENTATION_ROADMAP.md` R9.

## 2026-10-23 — Prompt wording for watch and keys (EXP-07)
- **Why:** in the live app the watch was found when worn, ~50 % on a table and almost never in the hand; keys were never found (EXP-06). Hypothesis tested: the prompt word is the cause.
- **What:** `OpenVocabDetector.set_target(..., prompts=[...])` and `openvocab_trials.py --prompts` (compare several words, or word combinations, on the same recorded frames). 7 recordings × 6 prompt sets with text YOLOE-26s.
- **Results:** watch: `watch` is best (worn 97.7 %); `wristwatch`, `wrist watch`, `analog watch` are worse; combinations equal `watch`; in hand 0 % with every prompt, so wording is **not** the cause. Keys: `bunch of keys` 99.7 % on a table vs 0 % for `key`, `keys`, `keychain`, `car key`; in hand 0 % with every prompt. No false sightings in the absent scenes (the "no keys" scene had no person in view).
- **Bug fixed:** `detection_trials.py` and `openvocab_trials.py` opened a new timestamped CSV for every run once the columns changed, instead of reusing today's file with matching columns. Data unaffected; 2 regression tests added (190 vision tests pass).
- **Decided (user-approved):** `decisions.md` D-14, the model gets `bunch of keys` when the user asks for keys (config `OPEN_VOCAB_PROMPTS`). 194 vision tests pass; live check with real keys: works (user report, informal).
- **Links:** `experiments/2026-10-23_prompt_wording/`.

## 2026-10-23 — R3: desk accuracy (EXP-06 Part B); YOLOE-26s text prompt chosen and integrated
- **What:** 8 desk recordings (watch on desk, watch on wrist, no watch, keys on desk, keys in hand, no keys, slippers on floor; one watch-on-desk recording redone and excluded), all three open-vocabulary models run on the same frames.
- **Results ("found", 0.25 + smoothing):** worn watch: text YOLOE-26s 99.7 %, prompt-free 76.4 %, YOLO-World 18.3 %; slippers: 99.7 % for all; keys: 0 % for all, even at 0.10; watch lying flat: 0 % (text YOLOE sees it at ~0.12 confidence in 73.4 % of frames); absent scenes: 0 % for all.
- **Decision:** D-13 decided — YOLOE-26s text prompt as the open-vocabulary fallback, threshold kept at 0.25.
- **Integration:** `module2_vision/detection/detector_factory.py` (`prepare_target_detector`, `HybridDetector`): the closed-set model keeps its 15 classes and keeps providing obstacles; other targets add the open-vocabulary model. `main.py` and `assistant.py` use it. New aliases: wristwatch → watch, eyeglasses/spectacles/specs → glasses. Config `OPEN_VOCAB_ENABLED`, `OPEN_VOCAB_MODEL_PATH`, `OPEN_VOCAB_CONFIDENCE_THRESHOLD`.
- **Verification:** 184 vision tests pass (7 new); real-model check: hybrid detection 32 ms per frame on the RTX 3050. Live hardware run with the app still to do.
- **Links:** `experiments/2026-10-23_open_vocab_comparison/`, `decisions.md` D-13, `paper_notes/claims_and_limitations.md` C13.

## 2026-10-23 — R3 started: open-vocabulary detectors benchmarked (EXP-06 Part A)
- **What:** compared three open-vocabulary detector families for objects outside the 15 classes: YOLOE-26 prompt-free (n, s), YOLOE-26 text-prompt (s, MobileCLIP2-B encoder) and YOLO-World v2 (s, CLIP ViT-B/32). Added `module2_vision/detection/open_vocab_detector.py` (same output format as `YOLODetector`), `tools/openvocab_benchmark.py` and `tools/openvocab_trials.py` (records one scene in memory and runs every model on the same frames).
- **Results (model-only, RTX 3050, 640 px):** YOLO-World v2 11.5–11.9 ms, YOLOE-26s text 16.9–20.1 ms, YOLOE-26s prompt-free 19.0–21.8 ms, YOLOE-26n prompt-free 30.9–32.7 ms (slower than s in both runs; cause not verified). Closed-set `best.pt` re-measured at 8.1–8.9 ms, which gives claim C10 its missing evidence.
- **Vocabulary:** the prompt-free models (4,584 names) include watch, key, slipper, glasses, earphone, charger, pen, medicine; not wallet, purse, remote control, headphones, eyeglasses, wristwatch.
- **Environment change:** installed the `clip` package (ultralytics/CLIP @ `7ffa84b`, plus ftfy, regex, wcwidth) into `.venv` and pinned it in `requirements.txt`. Weights and text encoders (~730 MB) are in `module2_vision/weights/` (gitignored). Two browser-downloaded files were corrupted despite correct sizes (CRC check); they were re-downloaded and verified, and the corrupted copies were kept as `*.corrupt` for the user to delete.
- **Pending:** Part B desk accuracy on watch / keys / slippers, then the model choice (D-13) and integration into `main.py` / `assistant.py`.
- **Tests:** 177 vision tests pass (13 new).
- **Links:** `experiments/2026-10-23_open_vocab_comparison/`, `decisions.md` D-13, `paper_notes/claims_and_limitations.md` C10, C12.

## 2026-10-22 — Result scripts no longer overwrite earlier results
- **What:** `module1_audio/evaluation/evaluate_models.py` writes each run to `evaluation/runs/results_<date_time>.csv` (opened in create-only mode); `results.csv` stays the 2026-09-16 run of record. `plot_results.py` and `semantic_eval.py` take `--results` (default unchanged). `module2_vision/tools/detection_trials.py` appends to `tools/results/detection_trials_<date>.csv` and records the settings with each row (thresholds, smoothing window/hits/hold, image size, device, duration); `detection_trials.csv` (2026-10-20) is frozen.
- **Why:** re-running either tool used to overwrite or mix in with the results cited in the paper drafts. Each row now records its config, as the reproducibility rule requires.
- **Effect on results:** none. The default `semantic_eval.py` output is identical to the saved 2026-10-21 re-run, and `plot_results.py` reproduces `comparison_graph_v2.png`.
- **Verification:** 4 new tests (`module2_vision/tests/test_detection_trials_output.py`); 96 + 164 tests pass.

## 2026-10-21 — Correction: RefCOCO expressions double-counted
- **What:** `refcoco_processed.csv` stacks the UNC and Google split files of RefCOCO, which contain the same 50,000 objects and expressions. Every expression appears twice: 133,784 unique, not 267,568. Absolute counts in the Stage-1 tables and figures (expressions, tokens, term frequencies) are doubled; percentages and averages are unaffected.
- **Why it matters:** these numbers appear in the paper drafts. A future grounding split must use one scheme only.
- **Action:** published tables and figures are left unchanged as historical records; the correction is documented. A decision is pending (D-09).
- **Links:** `dataset/README.md`, `decisions.md` D-09, `paper_notes/claims_and_limitations.md`.

## 2026-10-21 — Repository restructure + research/ set up as source of truth
- **What:** `module-1/` → `module1_audio/` (now also holds TTS); `module-3/` + `module-2/` → `module2_vision/` (`training/`, `notebooks/`); datasets → `data/raw/`, processed CSVs → `data/processed/`, YOLO dataset → `data/yolo_dataset/` (gitignored); `.env` at the project root; legacy per-module venvs deleted; superseded files moved to `_to_delete/`.
- **Research:** `research/` reorganised (`dataset/`, `experiments/`, `results/`, `baselines/`, `references/`, `paper_notes/`). Training curves and metrics, desk-trial data and the ASR results were frozen into `experiments/`. The semantic eval was re-run (it matches the published numbers exactly). `research/results/scripts/` was fixed (broken paths) and verified: all 8 tables regenerate byte-identical.
- **Verification:** 96 + 160 tests pass (baseline before the move: 77 + 179; 19 TTS tests moved with TTS). GPU inference with `best.pt` and stock `yolov8n.pt` from the new paths works. All notebook and `train.py` paths resolve.
- **Commits:** tag `pre-restructure-2026-10-21`, then `686e44e`, `317fd7f`, and the research/docs commits that follow.
- **Links:** `decisions.md` D-12.

## 2026-10-20 — Desk detection trials (fine-tuned vs stock)
- 10 runs, 15 s each. The new threshold + smoothing removes almost all flicker; a face behind the bottle is the main failure. The fine-tuned model beat stock in both face scenes → it stays the default.
- `experiments/2026-10-20_desk_detection_trials/` · commits `babe5f8`, `d667e5e`, `d93959f` · `decisions.md` D-05, D-06.

## 2026-10-20 — TTS fixes (every update spoken; first-word clipping)
- Switched to direct SAPI via comtypes; 250 ms lead silence; shorter phrases; rate 220.
- Commits `2cb24a6`, `f410694` · `decisions.md` D-10.

## 2026-10-20 — Manual end-to-end hardware test: 14/14 pass
- `experiments/2026-10-20_manual_hardware_test/` · commit `a224f98`.

## 2026-10-20 — Project audit and integration repairs
- One environment; voice↔vision bridge fixed (package-name clash); CPU/GPU device auto-selection; supported targets taken from the model; speech flooding fixed; session end conditions (arrival, timeout, cancel).
- `PROJECT_AUDIT.md` §5 · commits `684debe`, `dd01459` · `decisions.md` D-11.

## 2026-10-20 — YOLOv8n 15-class fine-tune finished (started 2026-10-14)
- 30 epochs on COCO train2014 (15 classes): P 0.567, R 0.467, mAP50 0.481, mAP50-95 0.331. The validation set probably overlaps the pretraining data, so these are optimistic.
- `experiments/2026-10-20_yolov8n_15class_finetune/` · commit `3fb09d9` · `decisions.md` D-03, D-04.

## 2026-10-14 — Fine-tuning started
- `train.py` built the YOLO dataset (46,688 train / 3,000 val images, hard links) and started training on the RTX 3050. It was interrupted and resumed on 2026-10-20 02:10.

## 2026-10-09 — Module 1 → Module 3 voice bridge; spatial guidance, obstacles, TTS (Milestones 4–5)
- Commit `6594dd8` · `decisions.md` D-07.

## 2026-10-05 — Real-time camera + YOLOv8n detection and target matching (Milestones 2–3)
- Stock 80-class `yolov8n.pt` at this point. Commits `c2dbee5`, `65fed60`.

## 2026-09-16 — ASR benchmark (3 models) + semantic evaluation; Stage-1 write-up
- Audio was re-recorded after the audit found silent WAVs. Sarvam had WER 2.0 %, Whisper Tiny 4.5 %, Whisper Base 6.5 %; semantic accuracy (complete) was 100/100/90 %.
- Stage-1 findings, paper draft, figures 01–13 and tables 01–08 were produced.
- `experiments/2026-09-16_asr_benchmark/`, `results/`, `paper_notes/` · commits `cdf59d2`, `6726126` · `decisions.md` D-01, D-02.

## 2026-09-16 — Stage-1 audit
- Found the benchmark WAVs were silent; fixed the parser's last-noun object choice; Google split `test` accepted as a valid label.
- `experiments/2026-09-16_asr_benchmark/STAGE_1_RESEARCH_AUDIT.md`.

## 2026-09-15 — Notebooks 01–03: inspection, preprocessing, EDA
- 604,906 COCO boxes kept (1 invalid removed); RefCOCO expressions deduplicated per split scheme; 14 EDA plots and 3 tables.
- `dataset/` (snapshots dated 2026-09-15/16).

## 2026-09-15 — ASR pilot, 2 models — INVALID
- Both models scored 0 % because the WAVs were silent. `experiments/2026-09-15_asr_pilot_2model/`.

## 2026-09-15 — Project set up
- Commit `df8aa7e`.

## 2026-09-14 — Datasets downloaded
- COCO 2014 train images + train/val annotations; RefCOCO (UNC + Google split files). `dataset/README.md`.
