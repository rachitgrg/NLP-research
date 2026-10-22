# To do before writing / submitting

Engineering milestones are in `IMPLEMENTATION_ROADMAP.md` (R1–R8). This list covers what the paper needs.

## Evidence gaps
- [ ] Clean detector evaluation on COCO val2017: fine-tuned vs stock, same 15 classes, plus FPS (R2; needs the val2017 download → ask first).
- [ ] Re-measure inference speed with a saved script and log (claim C10).
- [ ] Re-run and save the COCO "bottle recall with a person present" analysis, or drop the claim (C11).
- [ ] Decide on the RefCOCO split scheme (D-09). Recompute RefCOCO statistics on unique expressions and add them as a dated new table (don't overwrite table_05..07).
- [ ] A larger ASR test: more speakers, more languages, noise. If done, record it as a new experiment.
- [ ] A user study or at least scripted room trials: success rate and time to find (R7).

## Writing
- [ ] Decide on the title / framing (grounding vs detection) — `outline.md`.
- [ ] Update fig_01 to the current architecture (it shows Stage 1 only).
- [ ] Verify every entry in `../references/references.bib`; add related work.
- [ ] Update the counts in `PAPER_RESULTS_DISCUSSION.md` to the corrected RefCOCO numbers (keep a note of the change).

## Housekeeping
- [ ] Review `_to_delete/` (repo root) and delete or restore.
- [x] Make `evaluate_models.py` and `detection_trials.py` write dated result files instead of overwriting or appending to one file (2026-10-22).
