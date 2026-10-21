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
| C10 | Model inference ≈ 8.8 ms/frame on an RTX 3050 | ❌ no saved log | `PROJECT_AUDIT.md` §7 — re-measure with a script |
| C11 | Stock model recalls bottles better than fine-tuned on COCO photos with a person (47 % vs 40 %) | ❌ no saved artifact | `results/MANIFEST.md` |

## Must NOT be claimed (current state)
- **Audio-visual grounding / referring-expression comprehension.** The system detects fixed classes; attributes such as color and position are parsed but ignored (`decisions.md` D-08).
- **Distance or navigation accuracy.** "Reached" is a box-area heuristic (≥ 22 % of the frame), and left/right is in image space (D-07). Nothing has been measured.
- **Benefit for visually impaired users.** There has been no user study; all tests were done by the developer at a desk.
- **Open-vocabulary detection.** Objects outside the 15 classes (watch, keys, slippers…) are rejected.
- **Offline operation.** Speech recognition needs internet (Sarvam).

## Limitations to state in the paper
1. The ASR benchmark has n = 10, one speaker, English only, quiet room, and no significance testing.
2. Detector validation likely overlaps COCO pretraining data, and there is no clean val2017 comparison yet (roadmap R2).
3. Hardware evaluation has one tester, a laptop webcam, and a desk setup with no forward-facing camera.
4. A face or person directly behind the target sharply reduces detection (15–77 % of frames found).
5. Obstacle logic assumes a forward-facing camera and lists classes the 15-class model cannot output (bench, refrigerator, toilet).
6. Speech output is synchronous: an emergency message cannot interrupt an utterance already in progress (~1.7 s).
7. Fine-tuning dropped 65 COCO classes, some of them relevant (remote, clock, keyboard, handbag).
8. RefCOCO statistics in Stage-1 drafts double-count expressions (corrected 2026-10-21).
