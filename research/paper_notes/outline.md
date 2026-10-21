# Paper outline (draft)

Working title from Stage 1: *Multimodal Audio-Visual Grounding Framework for Real-Time Assistive Navigation*. ⚠ The current system does detection, not grounding (`../decisions.md` D-08). Either retitle (e.g. "voice-driven object finding with real-time spoken guidance") or implement grounding (roadmap R3/R6) before submitting.

1. **Introduction:** the problem of finding everyday objects without sight; contributions (only ✅ claims from `claims_and_limitations.md`).
2. **Related work:** assistive vision apps; referring-expression comprehension (RefCOCO); open-vocabulary detection; multilingual ASR. → `../references/`
3. **System:**
   - 3.1 Voice: Sarvam STT (translation to English) + rule parser (D-01, D-02)
   - 3.2 Vision: YOLOv8n 15-class model, threshold + temporal smoothing (D-03…D-06)
   - 3.3 Guidance: 5 zones, arrival heuristic, obstacle corridor, speech scheduling (D-07, D-10)
   - Figures: architecture (update fig_01, which covers Stage 1 only)
4. **Data:** COCO 2014 / RefCOCO statistics (use corrected counts), the 15-class subset, the split (`../dataset/README.md`)
5. **Experiments:**
   - 5.1 ASR pilot (EXP-02)
   - 5.2 Detector fine-tune (EXP-03) + clean val2017 baseline (**to do, R2**)
   - 5.3 Desk detection trials (EXP-05)
   - 5.4 End-to-end functional test (EXP-04); user study (**to do, R7**)
6. **Discussion:** occlusion by faces; closed vocabulary; image-space directions.
7. **Limitations:** from `claims_and_limitations.md`.
8. **Conclusion and future work:** open vocabulary (R3), attribute-aware selection (R6), distance cues and a user study (R7).
