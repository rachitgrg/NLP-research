# Design decisions

One entry per decision: what was decided, why, the evidence, and the date. If a decision changes, mark the old entry **Superseded by D-xx** and add a new one; don't edit history away. "Rationale not recorded" means the reason was never written down. Please fill it in, don't guess.

| ID | Decision | Date | Status |
|---|---|---|---|
| D-01 | Sarvam Saaras v3 as the runtime speech recogniser | 2026-09-16 | **Superseded by D-15** (2026-10-23) |
| D-02 | Rule-based spaCy parser with dependency-based object choice | 2026-09-16 | Active |
| D-03 | Detector restricted to 15 indoor COCO classes | 2026-10-14 | Active |
| D-04 | Detector split: image-level 80/20 of COCO train2014, seed 42, val ≤ 3,000 | 2026-10-14 | Active (known leakage, see D-04) |
| D-05 | Fine-tuned 15-class model is the app default | 2026-10-20 | Active, pending R2 |
| D-06 | Target-specific threshold 0.25 + temporal smoothing | 2026-10-20 | Active |
| D-07 | Guidance: 5 image-space zones; "reached" = box ≥ 22 % of frame | 2026-10-09 | Active (heuristic) |
| D-08 | Scope: closed-set detection now; no referring-expression grounding yet | 2026-10-20 | Active |
| D-09 | RefCOCO: use one split scheme only | 2026-10-21 | **Open** |
| D-10 | Speech output via Windows SAPI (comtypes), pyttsx3 elsewhere | 2026-10-20 | Active |
| D-11 | Supported targets = classes of the loaded model | 2026-10-20 | Active |
| D-12 | One environment (root `.venv`) and one repository layout | 2026-10-21 | Active |
| D-13 | Open-vocabulary fallback: YOLOE-26s text prompt, alongside the closed-set model | 2026-10-23 | Active |
| D-14 | Prompt map: "keys"/"key" → model prompt `bunch of keys` | 2026-10-23 | Active (approved by the user 2026-10-23) |
| D-15 | On-device speech recognition (faster-whisper, translate) replaces Sarvam; Sarvam opt-in and paper baseline | 2026-10-23 | Active; model size decided 2026-10-27: large-v3 + language hint (EXP-08 run 1) |
| D-16 | Fully offline by default (project USP); laptop prototype for the paper, phone/glasses later | 2026-10-23 | Active (user direction) |
| D-17 | Spoken replies in the user's language: phrase table + installed voice, else English | 2026-10-23 | Active; Hindi table is a draft pending native review |
| D-18 | Paper scope: evaluate Hindi and English only; Sarvam baseline only within its free credit | 2026-10-25 | Active (user direction) |

---

### D-01 · Sarvam Saaras v3 for speech recognition
- **Superseded by D-15 (2026-10-23):** the app is offline by default. Sarvam remains an opt-in backend and the cloud reference in EXP-08.
- **Decision:** the app uses Sarvam Saaras v3 (`mode="translate"`, automatic language detection), so any supported language arrives as English text. faster-whisper stays for evaluation and as a possible offline fallback.
- **Why:** it had the lowest WER in the pilot (2.0 % vs 4.5 % for Whisper Tiny and 6.5 % for Whisper Base), and speech-to-English translation happens in one call.
- **Trade-off:** it needs internet, and its latency includes the network (0.56 s mean vs 0.44 s for Whisper Tiny).
- **Evidence:** `experiments/2026-09-16_asr_benchmark/`. Hindi → correct target was checked manually in `experiments/2026-10-20_manual_hardware_test/`.

### D-02 · Rule-based parser
- **Decision:** spaCy `en_core_web_sm` rules extract intent, object and attributes. The object is the noun with an `nsubj`/`dobj`/`ROOT` dependency, not the last noun.
- **Why:** with the last-noun rule, "Where is my phone on the table?" gave `table` (`experiments/2026-09-16_asr_benchmark/STAGE_1_RESEARCH_AUDIT.md` §2.1).
- **Limits:** it picks one object and doesn't handle conjunctions. Attributes are extracted but not yet used downstream.

### D-03 · 15 detector classes
- **Decision:** person, chair, dining table, couch, bed (obstacles); bottle, cup, bowl, laptop, cell phone, book, tv, backpack, umbrella, suitcase (find targets). Source: `module2_vision/training/train.py` `TARGET_CLASSES`.
- **Why (as recorded in code comments):** these are indoor assistive-navigation objects; "cell phone" is marked critical for blind users. *Why these 15 and not, e.g., remote, clock, keyboard, handbag: rationale not recorded.*
- **Cost:** 65 COCO classes are dropped, including some useful ones (`PROJECT_AUDIT.md` §6).

### D-04 · Detector train/val split
- **Decision:** an image-level 80/20 split of the train2014 images that contain the 15 classes (`RANDOM_SEED = 42`). Val is capped at 3,000 images because the full val set (~11,700) caused a numpy memory error in `ap_per_class` (`train.py` comment). The Ultralytics training seed is `0`.
- **Known problem:** the pretrained base saw COCO train2017 ⊃ train2014, so val is probably not held out. See `experiments/2026-10-20_yolov8n_15class_finetune/` caveat 1.

### D-05 · Fine-tuned model as default
- **Decision:** `module2_vision/training/runs/train/weights/best.pt` is the default `YOLO_MODEL_PATH`.
- **Why:** in desk trials it found the bottle more often than stock `yolov8n.pt` in both face-occlusion scenes (48.2 % vs 27.5 %, 86.5 % vs 81.4 % found) (`experiments/2026-10-20_desk_detection_trials/`).
- **Pending:** a clean comparison on COCO val2017 (roadmap R2). If stock wins there, revisit, since its 80-class vocabulary is worth more.

### D-06 · Threshold and smoothing
- **Decision:** the requested class is accepted from confidence 0.25 (others stay at 0.40). The target counts as found after 2 hits in 5 frames and is held for 0.5 s (`module2_vision/config/config.py`, `matching/temporal_filter.py`).
- **Why:** at 0.40 the bottle flickered (≈ 70 % of frames near a person). With the new pipeline, lost events dropped from 22 → 0 at 2 m and from 9 → 0 with a person behind, with no false positives (`experiments/2026-10-20_desk_detection_trials/`).

### D-07 · Spatial guidance
- **Decision:** 5 horizontal zones from the box centre x (boundaries at 20/40/60/80 % of width). "Reached" = box area ≥ 22 % of the frame, confirmed over 5 frames; the same threshold applies to every class.
- **Why:** the simplest cue that works without depth. **Not measured.** Left/right is correct only if the camera faces the user's direction. Don't claim distance accuracy.

### D-08 · Scope: detection, not grounding
- **Decision:** the current system is a closed-set detector plus a rule parser. RefCOCO is used for EDA only. "The red bottle on the left" is not resolved, because attributes are ignored.
- **Why:** this is what's built. A detector returns every instance of fixed classes; grounding needs a text-conditioned model (`PROJECT_AUDIT.md` §6). Roadmap R3 (open vocabulary) and R6 (attribute-aware selection) are the next steps.
- **Paper implication:** don't describe the current results as audio-visual grounding.

### D-09 · RefCOCO split scheme (open)
- **Finding (2026-10-21):** the UNC and Google files cover the same 50,000 objects and expressions, and the processed CSV holds each expression twice (`dataset/README.md`).
- **Decision needed before any RefCOCO experiment:** use UNC (testA/testB, the common benchmark choice) or Google, never both. Stage-1 counts should be reported as 133,784 unique expressions.

### D-10 · Speech output engine
- **Decision:** Windows uses SAPI5 directly via comtypes (synchronous); other platforms use pyttsx3 with a fresh engine per utterance. Each message is preceded by 250 ms of silence. Rate is 220 wpm. An identical message repeats only after 4 s.
- **Why:** a reused pyttsx3 engine in a worker thread spoke only once; later calls returned in 0.07 s with no sound. Bluetooth headsets clipped the first word (`PROJECT_AUDIT.md` §5 #8; commits `2cb24a6`, `f410694`).

### D-11 · Supported targets come from the model
- **Decision:** the target check uses `detector.class_names` of the loaded model, not the COCO-80 list.
- **Why:** with the 15-class model, "clock" passed the COCO-80 check and was searched for forever (`PROJECT_AUDIT.md` §5 #4).

### D-12 · Environment and layout
- **Decision (2026-10-20):** one root `.venv` with voice and vision dependencies and CUDA torch.
- **Decision (2026-10-21):** layout `module1_audio/` (all sound: ASR, parser, TTS), `module2_vision/` (detection, guidance, training, notebooks), `research/` (paper source of truth), and a gitignored `data/` at the root with a `DATA_DIR` override.
  - Data sits at the root because both modules and the notebooks use it, raw data stays separate from processed data, and nothing large can be committed by accident.
- **Interface kept:** detection dict `{"class_name", "class_id", "confidence", "bbox": [x1, y1, x2, y2]}` (pixels).

### D-13 · Open-vocabulary fallback
- **Decision (2026-10-23, after EXP-06 Part B):** targets outside the closed-set classes are searched with **YOLOE-26s with text prompts** (`yoloe-26s-seg.pt` + MobileCLIP2-B), run together with the closed-set model (`HybridDetector`), which keeps providing people and furniture for obstacle warnings. Confidence threshold 0.25 (as for closed-set targets). Configurable: `OPEN_VOCAB_ENABLED`, `OPEN_VOCAB_MODEL_PATH`, `OPEN_VOCAB_CONFIDENCE_THRESHOLD`.
- **Why:** on the same recorded frames it found a worn watch in 99.7 % of frames vs 76.4 % (prompt-free YOLOE) and 18.3 % (YOLO-World), tied on slippers (99.7 %), with no false sightings in two absent scenes. It accepts any word, while the prompt-free vocabulary lacks e.g. wallet and remote control. Speed ≈ 15–20 ms alone, ≈ 32 ms together with the closed-set model.
- **Not changed:** threshold 0.10 would catch a watch lying flat (73.4 % of frames) but false-positive evidence is only two scenes; revisit with more absent/cluttered scenes. Keys are not found by any tested model.

#### History (before Part B)
- **Decided so far (2026-10-23):** keep the closed-set `best.pt` for its 15 classes (fastest: 8–9 ms; D-05). Targets it cannot detect go to an open-vocabulary model run through the installed Ultralytics package, behind the same `detect(frame)` interface (`module2_vision/detection/open_vocab_detector.py`).
- **Candidates and Part A evidence** (`experiments/2026-10-23_open_vocab_comparison/`): all run at ≥ 30 FPS model-only on the RTX 3050. Prompt-free YOLOE needs no text encoder but only knows its 4,584 names (watch, key, slipper yes; wallet, remote control no). Text-prompt YOLOE and YOLO-World accept any word but need a text encoder (254 MB / 354 MB) and the `clip` package.
- **Still to decide:** which model, from desk accuracy on watch / keys / slippers (EXP-06 Part B). Speed alone does not decide it.
- **Why not Grounding DINO / OWLv2:** heavier `transformers` dependency and a few FPS on 6 GB (roadmap R3 alternatives); not tested.

### D-14 · Prompt map for the open-vocabulary model
- **Decision (2026-10-23, approved by the user):** when the user asks for "keys" or "key", the open-vocabulary model gets the prompt `bunch of keys`; speech still uses the user's word ("Keys, left."). Other targets keep their own word as the prompt ("watch" is already the best watch prompt). Implemented as `OPEN_VOCAB_PROMPTS` in `module2_vision/config/config.py` (env format `target=prompt1|prompt2;…`), applied in `detection/detector_factory.py`; mapped words the model can't search for fall back to the target word.
- **Evidence:** `experiments/2026-10-23_prompt_wording/` (EXP-07): keys on a table 99.7 % found with `bunch of keys` vs 0 % with `key`, `keys`, `keychain`, `car key`; 0 % false sightings in a "no keys" scene.
- **Open risks:** one recording per scene; keys held in the hand still 0 %; false positives with a person in view untested for this prompt.

### D-15 · On-device speech recognition
- **Decision (2026-10-23):** the app recognises speech on the laptop with faster-whisper (`task="translate"`: any language → English text, so the parser of D-02 is unchanged). This is `ASR_BACKEND=whisper`, the default, in `module1_audio/speech/whisper_stt.py`, chosen through `speech/transcriber_factory.py`. Settings live in `speech/asr_config.py`:
  - `ASR_MODEL_SIZE` (default `large-v3`, provisional);
  - `ASR_LANGUAGE` (the user's language; default `auto`);
  - device and compute type (auto: GPU `int8_float16`, CPU `int8`).
- Sarvam (`ASR_BACKEND=sarvam`) needs `OFFLINE_MODE=false` and stays the cloud reference for EXP-08.
- **Why:** offline operation is the project's USP (D-16). Whisper is the only offline candidate that translates to English in one step, it is MIT-licensed, and it was already a dependency (EXP-02).
- **Rejected:**
  - `large-v3-turbo`: not trained for translation.
  - SeamlessM4T / MMS: CC-BY-NC, which blocks the final product.
- **Still open:** the model size and whether Indic languages are routed to AI4Bharat IndicConformer + IndicTrans2 (MIT). EXP-08 decides, by object accuracy per language with ≤ 2 s latency (`experiments/2026-10-23_multilingual_offline_asr/`).
- **Evidence so far:** English smoke test only, 10/10 objects with tiny/base on the GPU at ~0.1 s (EXP-08 smoke). No multilingual evidence yet.
- **Update 2026-10-27 (EXP-08 run 1, 120 Hindi + English clips, 3 speakers):** large-v3 90 % object accuracy in Hindi and 90.0–91.7 % in English; tiny/base only 33 %/55 % in Hindi. With the language given, large-v3 has a median of 0.55 s and a maximum of 1.73 s, which is within the 2 s rule. Provisional choice: large-v3 (already the `ASR_MODEL_SIZE` default) with `ASR_LANGUAGE` set by the user. Still open: `small`/`medium`, the Sarvam reference, and noisy conditions.
- **Decided 2026-10-27 (same day, all sizes + Sarvam):** by the EXP-08 rule, **large-v3 with the language given** for Hindi and English. English: 90.0 % within 2 s (auto 91.7 % has a 2.22 s clip). Hindi: tied with medium at 90.0 %. Fallback for smaller devices: medium+hint (1.5 GB; 90.0 % HI, 86.7 % EN, 0.35 s). The cost of going offline: 5–6 points below Sarvam (95.8 %). The AI4Bharat route is not pursued for now (Hindi already ties across sizes; would need an install). Re-check with the noisy clips.

### D-16 · Offline by default; laptop prototype for the paper
- **Decision (2026-10-23, user direction):**
  - Every session runs without internet. `OFFLINE_MODE` (default on) sets `YOLO_OFFLINE`, `HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE` before model libraries load, which stops Ultralytics' online check, telemetry and auto-update.
  - Missing model files fail fast (`module2_vision/tools/offline_check.py` is the pre-flight).
  - The paper uses the laptop prototype. The phone/glasses product is future work, supported by CPU-only measurements (EXP-10) rather than a rewrite.
- **Why:** cloud assistants (e.g. Gemini) need a connection. Offline also means privacy (no audio or video leaves the device), no network latency, and working without connectivity.
- **Caveat for the paper:** on-device object detection alone is not new. Google Lookout describes on-device vision, and several apps advertise offline detection. The claim must be the combination: offline multilingual spoken request + open-vocabulary target + continuous guidance to arrival, measured against a cloud baseline. Check related work before writing "first".
- **Not yet verified:** a full session with the network physically off (EXP-09).

### D-17 · Replies in the user's language
- **Decision (2026-10-23):**
  - Every spoken sentence lives in one table, `module2_vision/guidance/phrases.py`, with per-language object names. English is unchanged; Hindi is a draft.
  - Replies use the recogniser's language when that language has a table **and** an installed voice (Windows SAPI, `audio_feedback/tts_speaker.py` `voice_languages()`); otherwise they fall back to English. `REPLY_LANGUAGE` (`auto` or a code) overrides this.
  - No machine translation at runtime.
- **Why:** the guidance vocabulary is small and fixed (~15 templates), so reviewed tables are more reliable than MT and cost nothing at runtime.
- **Open:**
  - The Hindi table must be checked by a native speaker. It is worded so no verb agrees with the object's gender.
  - This machine has English voices only, so Hindi replies need a Windows Hindi voice (install approval).
  - Other languages need their own table and voice; until then they get English replies (a stated limitation).

### D-18 · Paper scope: Hindi and English; Sarvam within free credit
- **Decision (2026-10-25, user direction):**
  - The paper is the priority; the full product can follow later. Only work the paper's claims need is done now.
  - EXP-08 records and evaluates **Hindi and English only** (code-mixed Hinglish counts as Hindi, as spoken naturally). The user and the available speakers don't speak other languages. This replaces the EXP-08 target of "≥ 4 languages".
  - The Sarvam cloud baseline runs only while the account's free credit covers it; the project has no budget for paid calls.
- **Consequence for the paper:** the system supports any language Whisper/Sarvam accept, but only Hindi and English are measured. Other Indian languages are a stated limitation and future work; don't claim accuracy for them.
