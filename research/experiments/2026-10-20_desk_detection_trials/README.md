# EXP-05 · Desk detection trials: threshold + temporal smoothing, fine-tuned vs stock

| | |
|---|---|
| Date | 2026-10-20, 19:26–19:44 (timestamps in the CSV; commits `babe5f8`, `d93959f`) |
| Produced by | `module2_vision/tools/detection_trials.py --scene "<name>"` (at the time it appended one row per run to `module2_vision/tools/results/detection_trials.csv`; since 2026-10-22 new runs go to `detection_trials_<date>.csv` with their settings, and this file is frozen) |
| Data | `detection_trials_2026-10-20.csv` (frozen copy, 10 rows) |
| Setup | Laptop webcam at 30 FPS, target `bottle`, one 15 s run (~450 frames) per scene. Each frame goes through the model once; both pipelines are computed from the same frames. |
| OLD pipeline | target present if confidence ≥ 0.40 in that frame |
| NEW pipeline | target accepted from `TARGET_CONFIDENCE_THRESHOLD = 0.25` + `TargetTracker` (found after 2 of 5 frames, 0.5 s hold) |
| Models | fine-tuned `best.pt` (EXP-03) in 8 scenes; stock `yolov8n.pt` in 2 face scenes, run separately minutes later |

## Results (fine-tuned)

| Scene | Raw @0.40 | Raw @0.25 | Found OLD | Found NEW | Lost events OLD → NEW | Longest gap OLD → NEW (s) | Median conf |
|---|---|---|---|---|---|---|---|
| Alone, plain background | 100.0 % | 100.0 % | 100.0 % | 99.8 % | 0 → 0 | 0.00 → 0.02 | 0.96 |
| Person behind | 95.1 % | 98.9 % | 95.1 % | 99.8 % | 9 → 0 | 0.26 → 0.02 | 0.82 |
| Far, ~2 m | 81.6 % | 91.6 % | 81.6 % | 99.8 % | 22 → 0 | 0.35 → 0.03 | 0.58 |
| Dim light | 84.5 % | 87.1 % | 84.5 % | 95.1 % | 5 → 2 | 0.90 → 0.45 | 0.79 |
| Moving slowly, no face behind | 99.3 % | 99.6 % | 99.3 % | 99.8 % | 1 → 0 | 0.10 → 0.05 | 0.93 |
| Moving slowly, face behind | 69.8 % | 76.9 % | 69.8 % | 86.5 % | 12 → 2 | 2.10 → 1.40 | 0.70 |
| Held in front of face | 0.7 % | 15.3 % | 0.7 % | 48.2 % * | 1 → 6 | 7.62 → 2.15 | 0.21 |
| No bottle (false-positive check) | 0.0 % | 0.0 % | 0.0 % | 0.0 % | — | — | — |

\* Mostly the 0.5 s hold stretching sparse detections.

## Fine-tuned vs stock (face scenes)

| Scene | Model | Raw @0.25 | Found NEW | Median conf |
|---|---|---|---|---|
| Held in front of face | fine-tuned | 15.3 % | 48.2 % | 0.21 |
| Held in front of face | stock | 6.7 % | 27.5 % | 0.17 |
| Moving slowly, face behind | fine-tuned | 76.9 % | 86.5 % | 0.70 |
| Moving slowly, face behind | stock | 53.9 % | 81.4 % | 0.43 |

## Conclusions
- The NEW pipeline removes nearly all flicker when the model sees the target at all, and it produced no false bottles.
- A face directly behind the bottle is the main failure. This is a model limitation; thresholds and smoothing don't fix it.
- At this desk the fine-tuned model beat stock in both face scenes, so it stays the default (`decisions.md` D-05).

## Caveats
One tester, one webcam, one bottle, one run per scene. The stock and fine-tuned runs were separate, so hand position and lighting were not identical. Treat the results as indicative, not as a benchmark.
