# EXP-04 · Manual end-to-end hardware test (functional pass/fail)

| | |
|---|---|
| Date | 2026-10-20 (commit `a224f98`; source: `research/PROJECT_AUDIT.md` §7b) |
| Setup | Windows laptop, RTX 3050, built-in webcam (index 0), Bluetooth headset mic + output, root `.venv`, code at commit `dd01459`, fine-tuned `best.pt` |
| Procedure | One tester ran a 14-item checklist with `main.py` / `assistant.py` |

| # | Check | Result |
|---|---|---|
| 1 | Environment + CUDA | Pass |
| 2 | Module 1 tests (77) | Pass |
| 3 | Module 3 tests (152 at the time) | Pass |
| 4 | Live camera + YOLO detection | Pass |
| 5 | Typed target matching (found / not found) | Pass |
| 6 | Direction guidance (left / centre / right) | Pass |
| 7 | Arrival stop | Pass |
| 8 | ENTER cancellation | Pass |
| 9 | 15 s timeout | Pass |
| 10 | Unsupported targets (watch, clock) | Pass |
| 11 | Obstacle warnings without queue flooding | Pass |
| 12 | English voice → target → guidance | Pass |
| 13 | Hindi voice → correct target | Pass |
| 14 | Failure handling (silence, unclear speech, unsupported request) | Pass |

## Conclusion
The integrated system works end to end at a desk.

## Caveats
This was a functional check by one tester (the developer). It is **not** a usability study, an accuracy benchmark, or a test with visually impaired users. Do not cite it as evidence of navigation success rate.
