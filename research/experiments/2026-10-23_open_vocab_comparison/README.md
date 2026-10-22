# EXP-06 · Open-vocabulary detectors: speed, vocabulary, desk accuracy (roadmap R3)

**Status:** complete (2026-10-23). Part A: speed + vocabulary. Part B: desk accuracy on watch / keys / slippers. **Outcome:** YOLOE-26s text prompt chosen as the open-vocabulary fallback (`decisions.md` D-13) and integrated into the app.

| | |
|---|---|
| Date | 2026-10-22 (setup, downloads) – 2026-10-23 (benchmark) |
| Question | Which open-vocabulary detector should find objects outside the 15 closed-set classes (watch, keys, slippers, …), and at what speed? |
| Candidates | YOLOE-26 prompt-free `yoloe-26n-seg-pf.pt`, `yoloe-26s-seg-pf.pt` (built-in vocabulary); YOLOE-26 text-prompt `yoloe-26s-seg.pt` (+ MobileCLIP2-B text encoder `mobileclip2_b.ts`); YOLO-World v2 `yolov8s-worldv2.pt` (+ CLIP ViT-B/32) |
| Reference | closed-set default `module2_vision/training/runs/train/weights/best.pt` (EXP-03) |
| Code | `module2_vision/detection/open_vocab_detector.py` (wrapper with the YOLODetector output format), `module2_vision/tools/openvocab_benchmark.py` (Part A), `module2_vision/tools/openvocab_trials.py` (Part B) |
| Environment | root `.venv`: Python 3.13.1, torch 2.11.0+cu128, ultralytics 8.4.172, `clip` @ ultralytics/CLIP `7ffa84b` (installed 2026-10-22); NVIDIA GeForce RTX 3050 6GB Laptop GPU |
| Weights | Ultralytics assets release `v8.4.0` and OpenAI CLIP (`ViT-B-32.pt`, SHA-256 `40d36571…950af`, verified). All in `module2_vision/weights/` (gitignored). Every file was checked by zip CRC (and SHA-256 for CLIP) before use: two browser downloads were corrupted despite correct sizes and were re-downloaded. |

## Part A · Speed (model-only) and vocabulary

**Command (from `module2_vision/`):** `..\.venv\Scripts\python tools\openvocab_benchmark.py`, then the same with `--models` in reverse order to check ordering effects.
**Settings:** one image (`COCO_train2014_000000000714.jpg`, contains 2 bottles + 2 cups), 640 px, CUDA, conf 0.25, target `bottle`, 10 warm-up + 100 timed runs, `torch.cuda.synchronize()` around each call. Time = `detect()` call (preprocess + inference + NMS + conversion), **excluding** camera capture and drawing.
**Data:** `openvocab_benchmark_2026-10-23.csv` (frozen copy; 10 rows = 2 runs × 5 models).

| Model | Kind | Median ms (run 1 / run 2) | p90 ms (run 1 / run 2) | ≈ FPS (model only) | Time to set a target | Bottle confidences on the test image |
|---|---|---|---|---|---|---|
| `best.pt` (fine-tuned, 15 classes) | closed-set | 8.07 / 8.89 | 9.57 / 11.15 | 112–124 | — | 0.42, 0.32 |
| `yolov8s-worldv2.pt` | text prompt (CLIP) | 11.47 / 11.88 | 12.90 / 13.78 | 84–87 | 1.9 / 4.5 s (CLIP load) | 0.50, 0.39, 0.32 |
| `yoloe-26s-seg.pt` | text prompt (MobileCLIP2-B) | 16.92 / 20.08 | 20.23 / 22.21 | 50–59 | 0.44 / 0.41 s | 0.76, 0.70, 0.26 |
| `yoloe-26s-seg-pf.pt` | prompt-free | 18.96 / 21.75 | 21.82 / 25.09 | 46–53 | 0 | 0.76, 0.70 |
| `yoloe-26n-seg-pf.pt` | prompt-free | 30.88 / 32.72 | 33.90 / 35.09 | 31–32 | 0 | 0.48, 0.42 |

Observations:
- All candidates are well above the R3 acceptance bar of ≥ 10 FPS (model only).
- The **nano** prompt-free model is **slower** than the small one in both runs (order reversed between runs, so not a warm-up effect). Cause not verified; a plausible explanation is post-processing over its 4,584-class head.
- "Time to set a target" is paid once per search (text encoding); for YOLO-World it includes loading CLIP.
- The bottle confidences are on a single image and say nothing about accuracy; Part B measures that.
- This also re-measures the closed-set speed that claim C10 cited without a saved log: 8.1–8.9 ms median.

### Vocabulary coverage (prompt-free models)
**Data:** `vocab_coverage_2026-10-23.txt`. Produced by the snippet at the top of that file, run from `module2_vision/` with the matching rules of `open_vocab_detector.py` (exact match after lowercasing, `-`/`_` = space, plain "-s" plural; no "-es" rule so "glasses" never becomes "glass").

- Both prompt-free models have the same vocabulary: 4,585 raw names, 4,584 after normalisation.
- **In the vocabulary:** watch, key(s), slipper(s), sandal, flip flop, shoe, glasses, sunglasses, earphone, charger, pen, medicine, bottle, comb, towel, cup, mug.
- **Not in it:** wristwatch, eyeglasses, spectacles (synonyms → need an alias to "watch" / "glasses"), wallet, purse, remote control, headphones, pill, water bottle, cell phone, mobile phone. Those need a text-prompt model (or the closed-set model for "cell phone").

## Part B · Desk accuracy (user-run, 2026-10-23 03:27–03:34)
**Command (from `module2_vision/`):** `..\.venv\Scripts\python tools\openvocab_trials.py --target <object> --scene "<name>"` (default models `yoloe-26s-seg-pf.pt`, `yoloe-26s-seg.pt`, `yolov8s-worldv2.pt`).
**Method:** each scene recorded once for 10 s (~300 frames, 30 FPS laptop webcam) into memory (no images saved); every model runs on the same frames at record threshold 0.05. Metrics from `tools.detection_trials.analyze` (same definitions as EXP-05): "found" = target confirmed by `TargetTracker` (2-of-5, 0.5 s hold) at confidence ≥ 0.25. The closed-set `best.pt` was run on the same frames to record how often a person is in view. Settings are stored in each CSV row.
**Data:** `openvocab_trials_2026-10-23.csv` (frozen copy, 24 rows = 8 recordings × 3 models).
**Excluded run:** the first "watch on desk" recording (03:27:39–03:27:49, rows 1–3) — the tester reported setting it up wrongly and re-recorded it at 03:28:35. The rows are kept in the CSV and excluded from the table.

| Scene (target) | Person in view | Prompt-free YOLOE-26s: found / raw@0.25 / raw@0.10 | **Text YOLOE-26s**: found / raw@0.25 / raw@0.10 | YOLO-World v2 s: found / raw@0.25 / raw@0.10 |
|---|---|---|---|---|
| Watch on desk (redo) | 99.7 % | 0 / 0 / 0.7 % | 0 / 0.7 / **73.4 %** (median conf 0.12) | 0 / 0 / 0 % |
| Watch on wrist | 100 % | 76.4 / 36.2 / 37.2 % (lost 4) | **99.7 / 82.4 / 96.3 %** (lost 0) | 18.3 / 4.7 / 46.2 % (lost 2) |
| No watch | 100 % | 0 / 0 / 0 % | 0 / 0 / 0 % | 0 / 0 / 0 % |
| Keys on desk | 95.7 % | 0 / 0 / 0 % | 0 / 0 / 0 % | 0 / 0 / 0 % |
| Keys in hand | 100 % | 0 / 0 / 0 % | 0 / 0 / 0 % | 0 / 0 / 0 % |
| No keys | 13.6 % | 0 / 0 / 0 % | 0 / 0 / 0 % | 0 / 0 / 0 % |
| Slippers on floor | 0 % | 99.7 / 95.3 / 95.7 % (lost 0) | 99.7 / 100 / 100 % (lost 0) | 99.7 / 99.7 / 100 % (lost 0) |

Processing time on the recorded frames (ms/frame, model only): prompt-free 16.4–17.6, text YOLOE 14.4–16.2, YOLO-World 10.5–11.4.

### Conclusions
- **Text-prompt YOLOE-26s is the best candidate:** best on the worn watch (99.7 % found, no dropouts, vs 76.4 % and 18.3 %), tied on slippers, no false sightings in the two absent scenes.
- **A watch lying flat** is only seen at low confidence by text YOLOE (median 0.12; 73.4 % of frames ≥ 0.10). A 0.10 open-vocabulary threshold would catch it, and the absent scenes stayed at 0 % even at 0.10, but two scenes are too little false-positive evidence to lower the threshold (kept at 0.25; D-13).
- **Keys were not detected by any model**, even at 0.10, in either scene. The prompt word may matter ("keys" vs "key", "keychain"); untested. Keys remain a limitation.
- **Slippers** are easy for all three models.

## Integration (2026-10-23)
- `module2_vision/detection/detector_factory.py`: `prepare_target_detector()` keeps the closed-set `best.pt` for its 15 classes; any other target loads the open-vocabulary model (config `OPEN_VOCAB_MODEL_PATH`, default `yoloe-26s-seg.pt`) and runs it **with** the closed-set model (`HybridDetector`), so obstacle warnings keep working. Used by `main.py` and `assistant.py`.
- Aliases: wristwatch → watch, eyeglasses / spectacles / specs → glasses ("watch" is still never mapped to "clock").
- Real-model check (one COCO image, RTX 3050, 50 runs): "bottle" → closed-set only, 7.5 ms; "my wristwatch" → watch, hybrid, 32.0 ms (0.75 s one-off setup); "wallet" → hybrid, 32.8 ms (accuracy for wallet untested).

## Informal live observations (user, 2026-10-23, after integration; not measured)
App run `main.py --target watch|slippers --navigate --no-obstacles` with YOLOE-26s text prompt:
- Slippers: worked well.
- Watch worn on the wrist: detected most of the time (consistent with Part B, 99.7 %).
- Watch on the table: about 50 % of the time (Part B recording: 0 % at 0.25; confidence ~0.12, so it sits near the threshold and varies with angle and light).
- Watch held in the hand, not worn: detected only 1–2 times in about 2 minutes.
- Hypotheses (untested): the text embedding for "watch" matches the worn appearance it saw most in training (face + strap around a wrist); in the hand the face is partly covered or edge-on and the strap hangs loose. The prompt word may also matter ("watch" is also a verb; the alias currently maps "wristwatch" → "watch"). Next step: compare prompt words on the same recorded frames (roadmap R3 "Next").

## Caveats
- Part B: one tester, one desk, one webcam, one object of each kind, one 10 s recording per scene. Indicative, not a benchmark. False-positive evidence is two absent scenes only.
- Part A uses one image; speed on live 640×480 frames may differ slightly. Camera capture and drawing are excluded.
- Two benchmark runs per model only; no variance analysis beyond them.
- Vocabulary coverage says whether a name *can* be requested, not whether the object is detected reliably.
