# Claims and limitations

Every claim the paper might make, with the evidence behind it. Evidence labels (from `../PROJECT_AUDIT.md`): **[V]** verified by running code, **[M]** manual hardware test, **[R]** read from code or files, **[I]** inference. Only cite claims marked ✅.

## Claims

| # | Claim | Status | Evidence |
|---|---|---|---|
| C1 | On a 10-utterance pilot, Sarvam Saaras v3 reached 2.0 % WER vs 4.5 % (Whisper Tiny) and 6.5 % (Whisper Base) | ✅ as a pilot only | [V] `experiments/2026-09-16_asr_benchmark/` |
| C2 | The rule parser preserved intent/object/attributes on all 10 utterances for Sarvam and Whisper Tiny output (90 % complete for Whisper Base) | ✅ as a pilot only | [V] same, re-run 2026-10-21 |
| C3 | Spoken requests in English and Hindi led to the correct target being searched for | ✅ functional, n = 1 tester | [M] `experiments/2026-10-20_manual_hardware_test/` |
| C4 | The fine-tuned 15-class YOLOv8n reaches mAP50 0.481 on a COCO train2014 hold-out | ⚠ optimistic: likely pretraining overlap | [V]+[I] `experiments/2026-10-20_yolov8n_15class_finetune/` |
| C5 | Target-specific threshold + temporal smoothing removes most detection dropouts (e.g. 22 → 0 lost events at 2 m) | ✅ indicative, 1 run per scene | [M]+[V] `experiments/2026-10-20_desk_detection_trials/` |
| C6 | The fine-tuned model detects a bottle near a face more often than stock YOLOv8n | ⚠ 2 scenes, separate runs | [M] same |
| C7 | The end-to-end system (voice → detection → spoken direction → arrival stop) works on a laptop in real time | ✅ functional, desk only | [M] hardware test |
| C8 | RefCOCO: 133,784 unique expressions for 50,000 objects in 19,994 images; person = 50.17 % of expressions | ✅ (use 133,784, not 267,568) | [V] `dataset/README.md` |
| C9 | COCO train2014: 604,906 valid boxes; person = 30.6 % | ✅ | [R] notebook outputs |
| C10 | Closed-set model inference ≈ 8–9 ms/frame (model only, 640 px) on an RTX 3050 | ✅ re-measured 2026-10-23: 8.07 / 8.89 ms median over 2 × 100 runs, one image; excludes camera capture and drawing | [V] `experiments/2026-10-23_open_vocab_comparison/` (the original 8.8 ms of 2026-10-20 had no saved log) |
| C11 | Stock model recalls bottles better than fine-tuned on COCO photos with a person (47 % vs 40 %) | ❌ no saved artifact | `results/MANIFEST.md` |
| C12 | Open-vocabulary detectors run in real time on the RTX 3050: YOLO-World v2 s ≈ 12 ms, YOLOE-26 s ≈ 17–22 ms, YOLOE-26 n prompt-free ≈ 31–33 ms per frame (model only) | ✅ speed only, 2 runs, one image | [V] `experiments/2026-10-23_open_vocab_comparison/` |
| C13 | With the open-vocabulary fallback, the system detects objects outside its 15 classes: a worn watch in 99.7 % and slippers on the floor in 99.7 % of frames, with no false sightings in two absent scenes | ✅ indicative: 1 tester, 1 desk, one 10 s recording per scene; keys **not** found (0 %), a watch lying flat not found at 0.25 | [M]+[V] `experiments/2026-10-23_open_vocab_comparison/` Part B |
| C14 | Text-prompt YOLOE-26s beats prompt-free YOLOE-26s and YOLO-World v2 for a worn watch on identical frames (99.7 % vs 76.4 % vs 18.3 %) | ✅ one scene, one recording | [M]+[V] same |
| C15 | Prompt wording matters for open-vocabulary detection: keys on a table were found in 99.7 % of frames with "bunch of keys" vs 0 % with "key"/"keys"/"keychain"/"car key" on the same frames; for a watch, "watch" beat all alternatives | ✅ one recording per scene | [M]+[V] `experiments/2026-10-23_prompt_wording/` |
| C16 | Objects held in the hand (watch, keys) are not detected with any tested prompt | ✅ as a negative result, one recording each | [M]+[V] same |
| C17 | On-device Whisper (tiny/base, translate) extracted the correct object from all 10 English pilot utterances, at ~0.1 s per request on the RTX 3050 and 0.2–0.4 s on the laptop CPU | ✅ English pilot only (the EXP-02 clips, one speaker); says nothing about other languages | [V] `experiments/2026-10-23_multilingual_offline_asr/` (smoke), `experiments/2026-10-23_edge_feasibility/` |
| C18 | Without a GPU (Ryzen 5 7640HS), the closed-set detector runs at 22.6 ms/frame and the open-vocabulary YOLOE-26s at 73.6 ms/frame at 640 px (29.2 ms at 320 px) | ✅ speed only, model-only, one image; 320 px accuracy not measured | [V] `experiments/2026-10-23_edge_feasibility/` |
| C19 | The full assistant (voice request → offline speech recognition → detection → spoken guidance → arrival) runs with no network connection: with Airplane mode and Wi-Fi off, 14/14 spoken requests (Hindi + English) were heard as the right object, and in the final configuration 10/10 sessions reached the object (5 objects × 2 languages) | ✅ functional: one tester, one desk, laptop webcam + RTX 3050, one attempt per session; network-off confirmed by the tester (Airplane mode, `Test-NetConnection` = False) | [M]+[V] `experiments/2026-10-27_offline_end_to_end/` |
| C20 | Offline Whisper large-v3 (translate, on the RTX 3050) extracted the correct object from 90.0 % of Hindi and 91.7 % of English find-requests (3 speakers × 20 prompts per language), median 0.55 s per request with the language given; tiny/base/small reach only 33 %/55 %/82 % in Hindi | ✅ indicative: 3 speakers, quiet room, one microphone, n = 60 per language (±8 pp) | [V] `experiments/2026-10-23_multilingual_offline_asr/` run 1 (2026-10-27) |
| C21 | The offline recogniser is 5–6 points below the Sarvam cloud reference on the same clips (Sarvam 95.0 % EN / 96.7 % HI; large-v3+hint 90.0 % / 90.0 %); Sarvam was right on every clip Whisper got right | ✅ indicative: same conditions as C20; paired McNemar p = 0.016 uncorrected, 0.10 after Bonferroni ×6, so state it as a consistent gap, not as significant | [V] same, run 1 |
| C22 | Without a GPU (Ryzen 5 7640HS), the chosen recogniser is too slow for conversation: Whisper large-v3 7.5 s median per request, medium 3.6 s; only small is near real time (1.2 s) at 81.7 % Hindi / 88.3 % English object accuracy | ✅ one laptop CPU, the 120 EXP-08 clips, language given | [V] `experiments/2026-10-23_edge_feasibility/` (2026-10-27) |

## Must NOT be claimed (current state)
- **Audio-visual grounding / referring-expression comprehension.** The system detects fixed classes; attributes such as color and position are parsed but ignored (`decisions.md` D-08).
- **Distance or navigation accuracy.** "Reached" is a box-area heuristic (≥ 22 % of the frame), and left/right is in image space (D-07). Nothing has been measured.
- **Benefit for visually impaired users.** There has been no user study; all tests were done by the developer at a desk.
- **General open-vocabulary accuracy.** Only watch, keys and slippers were measured, at one desk (C13). Don't claim the system finds "any object"; keys failed, and untested words (wallet, glasses, …) have unknown accuracy.
- **Offline operation beyond C19's conditions.** It was shown on one laptop with an RTX 3050, at one desk. Without the GPU the chosen speech model is too slow (C22); don't claim offline operation on a phone or glasses.
- **"First offline assistant".** On-device object detection already exists (e.g. Google Lookout describes on-device vision; several apps advertise offline detection). Claim only the measured combination: offline multilingual request + open-vocabulary target + guidance to arrival.
- **Multilingual accuracy** beyond Hindi and English, or Hindi accuracy beyond C20's conditions (quiet room, 3 speakers). Don't claim parity with Sarvam: offline is 5–6 points lower (C21). Spoken Hindi replies: the phrases were reviewed by a native speaker and a Windows Hindi voice speaks them (2026-10-27, D-17), but they haven't been evaluated with users.

## Limitations to state in the paper
1. The ASR benchmark has n = 10, one speaker, English only, quiet room, and no significance testing.
2. Detector validation likely overlaps COCO pretraining data, and there is no clean val2017 comparison yet (roadmap R2).
3. Hardware evaluation has one tester, a laptop webcam, and a desk setup with no forward-facing camera.
4. A face or person directly behind the target sharply reduces detection (15–77 % of frames found).
5. Obstacle logic assumes a forward-facing camera and lists classes the 15-class model cannot output (bench, refrigerator, toilet).
6. Speech output is synchronous: an emergency message cannot interrupt an utterance already in progress (~1.7 s).
7. Fine-tuning dropped 65 COCO classes, some of them relevant (remote, clock, keyboard, handbag).
8. RefCOCO statistics in Stage-1 drafts double-count expressions (corrected 2026-10-21).
9. Open-vocabulary fallback: objects held in the hand were not detected with any prompt; keys needed the prompt "bunch of keys" (EXP-07); a watch lying flat is seen only at low confidence (~0.12); evidence comes from one desk with one object of each kind (EXP-06, EXP-07).
10. Multilingual evaluation covers Hindi and English only (D-18). Other Indian languages are supported by the recognisers but untested; the speakers available to the project don't speak them.
11. Speech recognition was measured in a quiet room only (D-19): 3 speakers, one laptop microphone, one take per request. Accuracy in background noise is unknown.
12. Arrival for small open-vocabulary objects uses per-object area thresholds tuned on one probe each (watch 3 %, keys 9 %; D-20); other small objects are untested.
