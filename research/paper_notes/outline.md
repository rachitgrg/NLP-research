# Paper outline (draft, updated 2026-10-27)

The previous outline (Stage 1, Sarvam cloud recognition, EXP-01–05 only) is in git history. This version reflects the system as of 2026-10-27: offline, Hindi + English, closed-set + open-vocabulary detection.

**Working title (to decide with the mentor):** *An Offline, Multilingual Voice-Driven Object Finder with Real-Time Spoken Guidance for Visually Impaired Users*.
⚠ Don't use "audio-visual grounding": the system detects named objects; it does not resolve referring expressions (`../decisions.md` D-08). The Stage-1 title used "grounding".

**Contribution to claim (only the measured combination, see `claims_and_limitations.md`):** a laptop prototype that takes a spoken request in Hindi or English, recognises it fully on-device, finds the object with a closed-set detector plus an open-vocabulary fallback, and guides the user to it by voice in the same language, with no network connection (C19). Don't claim "first offline assistant".

1. **Introduction**
   - Problem: finding everyday objects without sight; cloud assistants need a connection and mostly English.
   - Contributions: (a) offline multilingual pipeline (C19, C20); (b) hybrid closed-set + open-vocabulary detection (C13–C15); (c) evaluation of the offline/cloud accuracy gap (C21) and CPU feasibility (C18, C22).
2. **Related work:** assistive vision apps (Seeing AI, Lookout, Be My Eyes), open-vocabulary detection (YOLO-World, YOLOE), multilingual ASR (Whisper, Indic ASR), referring-expression comprehension (RefCOCO) as future work. → `../references/` (verify every entry)
3. **System** (architecture figure: update fig_01, which still shows Stage 1)
   - 3.1 Speech: faster-whisper large-v3, translate to English, language given (D-15); spaCy rule parser (D-02); target normalisation and aliases.
   - 3.2 Vision: fine-tuned YOLOv8n, 15 classes (D-03, D-05); YOLOE-26s text-prompt fallback for other objects (D-13), prompt map (D-14); target threshold + temporal smoothing (D-06).
   - 3.3 Guidance: 5 image-space zones, arrival by box area with a per-object threshold (D-07, D-20), confirmation over 5 frames, obstacle warnings, spoken replies in the user's language (D-17).
   - 3.4 Offline design: `OFFLINE_MODE`, pre-flight check, all models on disk (D-16).
4. **Data**
   - COCO 2014 subset for the 15 classes; split and its known leakage (D-04, `../dataset/README.md`).
   - EXP-08 speech set: 120 clips, 3 speakers × 20 requests × Hindi/English, quiet room (D-18, D-19).
   - RefCOCO statistics only if R6 is done (use the corrected counts, C8).
5. **Experiments**
   - 5.1 Offline speech recognition, Hindi + English (EXP-08): Whisper sizes ± language hint vs Sarvam (C20, C21). Table + McNemar.
   - 5.2 Detection: closed-set fine-tune (EXP-03, ⚠ optimistic until R2), desk trials and smoothing (EXP-05, C5), open-vocabulary comparison and prompt wording (EXP-06, EXP-07; C12–C16).
   - 5.3 End-to-end offline sessions (EXP-09, C19) with per-stage latency.
   - 5.4 Edge feasibility without a GPU (EXP-10, C18, C22).
6. **Discussion:** offline costs 5–6 points of recognition accuracy; the language hint matters for speed, not accuracy; small and flat objects (watch on a table, held objects) are the main detection gap; the arrival heuristic needs per-object sizes; GPU dependence.
7. **Limitations:** list 1–12 in `claims_and_limitations.md` (one tester, desk only, no user study, quiet room, 3 speakers, no distance measurement).
8. **Conclusion and future work:** watch-on-table improvement (R10), clean detector baseline (R2), attribute-aware selection (R6), user study and distance cues (R7), phone/glasses port (D-16).

## Experiments to cite, and their status
| Section | Experiment | Status |
|---|---|---|
| 5.1 | EXP-08 run 1 | ✅ done |
| 5.2 | EXP-03 | ⚠ needs R2 (val2017) for a clean number |
| 5.2 | EXP-05, EXP-06, EXP-07 | ✅ indicative (one desk) |
| 5.3 | EXP-09 | ✅ done (C19) |
| 5.4 | EXP-10 | ✅ speech part; detector part speed only |
| — | EXP-02 (cloud pilot, n = 10) | background only; superseded by EXP-08 for the paper |
