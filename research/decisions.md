# Design decisions

One entry per decision: what was decided, why, the evidence, and the date. If a decision changes, mark the old entry **Superseded by D-xx** and add a new one; don't edit history away. "Rationale not recorded" means the reason was never written down. Please fill it in, don't guess.

| ID | Decision | Date | Status |
|---|---|---|---|
| D-01 | Sarvam Saaras v3 as the runtime speech recogniser | 2026-09-16 | Active |
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

---

### D-01 · Sarvam Saaras v3 for speech recognition
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
