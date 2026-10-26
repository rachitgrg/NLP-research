# Project summary for mentor review (2026-10-27)

**Project:** an assistive object finder for visually impaired users. The user asks for an object by voice ("Where is my bottle?" / "मेरी बोतल कहाँ है?"); the laptop camera finds it, and the system guides the user to it with short spoken directions ("Bottle, left.") until it is within reach.

**Current state in one line:** a working laptop prototype that runs **fully offline**, understands **Hindi and English**, finds **15 trained object types plus open-vocabulary objects** (e.g. watch, keys), and replies in the user's language. It has been tested end to end with the network off.

Code: branch `feature/offline-multilingual`. Every number below comes from `research/` (experiment folders, `paper_notes/claims_and_limitations.md`).

---

## 1. How the system works

```
Microphone ─► Whisper large-v3, on-device (speech → English) ─► spaCy parser (which object?)
                                                                         │
Camera ─► YOLOv8n, 15 trained classes ──┐                                ▼
       └► YOLOE-26s, open vocabulary ───┴─► match the target ─► direction + distance cue
             (only for other objects)                                    │
                                    Windows voice (Hindi or English) ◄───┘
```

- **No internet needed:** all models are stored on the laptop, and a pre-flight script checks that every file is present.
- **Two detectors:** the fast fine-tuned model handles common objects (bottle, cup, phone, laptop, …). Anything else (watch, keys, slippers, …) goes to an open-vocabulary model that takes the object's name as a text prompt.
- **Guidance:** five left-to-right zones in the image, and "within reach" when the object's box is large enough for 5 frames in a row.

## 2. What has been done

| Phase | Dates | Work |
|---|---|---|
| Stage 1 | Sep 2026 | Speech pipeline and rule-based parser; ASR pilot (cloud Sarvam vs Whisper tiny/base, 10 clips); COCO + RefCOCO data cleaning and statistics |
| Detection + guidance | Oct 14–20 | YOLOv8n fine-tuned on 15 indoor COCO classes; spoken guidance and arrival; obstacle warnings; 14-item hardware test (14/14 pass); desk trials for threshold and frame smoothing |
| Open vocabulary | Oct 21–23 | Repository cleanup (one environment, tests); compared three open-vocabulary detectors, chose YOLOE-26s with a text prompt; prompt wording study ("bunch of keys" instead of "keys") |
| Offline + multilingual | Oct 23–27 | Replaced cloud speech recognition with on-device Whisper; offline mode; Hindi/English test set (3 speakers, 120 clips); compared every Whisper size with the cloud service; CPU-only timing; spoken replies in Hindi (phrases reviewed by a native speaker; Windows Hindi voice); offline end-to-end sessions |

Automated tests: 126 (speech module) + 220 (vision module), all passing. They mock the camera, models and voice, and fail if any test opens a network connection.

## 3. Main results (citable, with their limits)

| What was measured | Result | Limits |
|---|---|---|
| **Offline speech recognition, Hindi + English** (EXP-08) | Whisper large-v3 picked out the correct object in **90.0 % of Hindi** and **90.0–91.7 % of English** requests; 0.55 s per request with the language given. Smaller models are much worse in Hindi (tiny 33 %, base 55 %, small 82 %, medium 90 %). | 3 speakers, quiet room, one microphone, 60 clips per language (about ±8 points) |
| **Offline vs cloud** (EXP-08) | The cloud service (Sarvam) reached **95.8 %**, so going offline costs **5–6 points**. Sarvam was right on every clip Whisper got right, plus 7 more. | A consistent gap, but not significant after correcting for 6 comparisons (p = 0.10) |
| **Language hint** | Telling Whisper the language doesn't change accuracy but makes it ~40 % faster | Same set |
| **End-to-end, network off** (EXP-09) | Airplane mode and Wi-Fi off: **14/14 requests heard as the right object** (Hindi + English); in the final version **10/10 sessions reached the object** (bottle, phone, cup, worn watch, keys × 2 languages) | One tester, one desk, one attempt per session |
| **Open-vocabulary detection** (EXP-06/07) | Worn watch found in 99.7 % of frames, slippers 99.7 %, keys on a table 99.7 % with the prompt "bunch of keys" (0 % with "keys"); no false sightings in the empty scenes | One desk, one recording per scene |
| **Speed on the laptop GPU** (RTX 3050) | Closed-set detector 8–9 ms per frame; open-vocabulary 17–22 ms | Model-only timing, one image |
| **Without a GPU** (EXP-10) | The detectors still run (23 ms / 74 ms per frame), but Whisper large-v3 takes **7.5 s** per request; only the small model is near real time (1.2 s, 82 % Hindi) | One laptop CPU |
| **Fine-tuned detector** (EXP-03) | mAP50 0.481 on a COCO hold-out | ⚠ Optimistic: the hold-out probably overlaps the pretraining data. A clean test is planned (see 5) |

## 4. Bugs found during testing and fixed (also documented)

- **Watch/keys sessions never ended:** "directly in front of you" was spoken, but the session continued. Cause: small objects never fill 22 % of the frame (a worn watch peaked at 4 %, keys at 11 %), and the arrival message was spoken before arrival was confirmed. Fix: arrival size set per object, and the message only once arrival is confirmed. Re-test: 4/4 arrived.
- **8–10 s delay after speaking:** the speech model loaded after the request. It now loads before the prompt, so the reply waits ~1.2 s.
- **The Hindi voice wasn't used:** Windows installs it as a voice type the program didn't read. Fixed; Hindi replies now use the Hindi voice.

## 5. Plans (decided, not yet implemented)

| Plan | Why | Status |
|---|---|---|
| **Watch lying on a table** (R10) | A blind user looks for a watch on a table, not on the wrist; a flat watch currently scores 0 % (it is seen at only ~0.12 confidence). The cause is unclear: training data (mostly worn watches), the low camera angle, or the small size. | Proposed. First a proper test set (several watches, surfaces, two camera angles, empty scenes), then cheap fixes: camera looking down, lower threshold for watch, higher resolution, or "show it your watch once" (an example photo as the prompt). Retraining only if needed. |
| **Clean detector baseline** (R2) | The 0.481 mAP50 is likely optimistic | Needs the COCO val2017 download (~1 GB); compare fine-tuned vs stock on the same 15 classes |
| **Paper writing** | All evidence for the system, speech and offline sections is complete | Outline updated (`paper_notes/outline.md`); title to be decided |
| **Use colour and position words** (R6) | "The blue cup near the laptop" is parsed, but the attributes are ignored | Planned; evaluation on RefCOCO |
| **User study and distance cues** (R7) | Everything so far was tested by the developer at a desk | Planned: scripted rooms, success rate and time to find, a chest or phone camera |
| **Phone / glasses version** (D-16) | The final product target | Later. EXP-10 shows it would need the small speech model or a hardware accelerator |
| **Optional speed-up** | A faster CPU speech model (ONNX/INT8) | Optional; needs a package install |

Scope decisions already taken: only Hindi and English are evaluated (the available speakers); quiet room only; the cloud baseline only within its free credit.

## 6. What we must not claim (yet)

- Benefit for blind users: no user study yet.
- Distance or navigation accuracy: "within reach" is a box-size rule, not a measured distance.
- That it finds "any object": only watch, keys and slippers were measured; held objects and a watch lying flat fail.
- Offline operation on a phone: shown only on a laptop with a GPU.
- "Audio-visual grounding": the system detects named objects; it doesn't resolve phrases like "the cup on the left".
- "First offline assistant": on-device detection already exists elsewhere. We claim the measured combination.

## 7. Questions for the mentor

1. **Title and framing:** "offline multilingual object finder" (what was built and measured) instead of the Stage-1 "audio-visual grounding" title. Agree?
2. **Venue and length:** which conference or journal, and is a desk-level evaluation enough, or is a small user study needed before submission?
3. **Priority before writing:** fix the watch-on-table gap (R10), or run the clean detector baseline (R2) first?
