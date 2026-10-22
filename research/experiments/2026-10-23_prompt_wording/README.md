# EXP-07 · Prompt wording for the open-vocabulary detector (watch, keys)

| | |
|---|---|
| Date | 2026-10-23, 04:03–04:11 (timestamps in the CSVs) |
| Question | Does the prompt word explain the watch/keys failures in EXP-06 and the informal live test (watch found on the wrist, ~50 % on a table, almost never in the hand; keys never)? |
| Model | `yoloe-26s-seg.pt` (text prompt, MobileCLIP2-B), the open-vocabulary default (`decisions.md` D-13) |
| Command (from `module2_vision/`) | `..\.venv\Scripts\python tools\openvocab_trials.py --models yoloe-26s-seg.pt --target <watch|keys> --scene "<name>" --prompts <list>` |
| Prompts | watch: `watch`, `wristwatch`, `wrist watch`, `analog watch`, `watch+wristwatch`, `watch+wristwatch+wrist watch`. keys: `key`, `keys`, `keychain`, `bunch of keys`, `car key`, `key+keys+keychain`. `a+b` = both words given to the model at once; a hit for either counts. |
| Method | Each scene recorded once for 10 s (~300 frames, 30 FPS laptop webcam), in memory only; every prompt set runs on the same frames. Metrics from `tools.detection_trials.analyze` as in EXP-05/06; "found" = 0.25 threshold + `TargetTracker` (2-of-5, 0.5 s hold). Settings are stored in each row. |
| Code | `open_vocab_detector.py` `set_target(..., prompts=[...])`, `openvocab_trials.py --prompts` (added 2026-10-23) |
| Environment | as EXP-06 (root `.venv`, ultralytics 8.4.172, RTX 3050) |
| Data | 7 frozen CSVs, one per recording (`openvocab_trials_2026-10-23_HHMMSS.csv`, 6 rows each). They are separate files because of a file-naming bug in the trial tools (each run with the new `prompts` column opened a new file); fixed the same day, data unaffected. |

## Results ("found" % / raw rate at 0.25 / raw rate at 0.10; median confidence)

### Watch
| Prompt | On table | Worn | In hand | No watch |
|---|---|---|---|---|
| `watch` | 5.0 / 1.3 / 59.3 (0.12) | **97.7 / 70.8 / 89.7 (0.43)** | 0 / 0 / 0.3 | 0 / 0 / 0 |
| `wristwatch` | 0 / 0 / 7.0 | 92.0 / 49.5 / 76.1 (0.31) | 0 / 0 / 0 | 0 / 0 / 0 |
| `wrist watch` | 0 / 0 / 0 | 57.1 / 21.3 / 52.2 (0.17) | 0 / 0 / 0 | 0 / 0 / 0 |
| `analog watch` | 0 / 0 / 0 | 68.1 / 27.6 / 66.1 (0.18) | 0 / 0 / 0 | 0 / 0 / 0 |
| `watch+wristwatch` | identical to `watch` | identical to `watch` | identical | 0 |
| `watch+wristwatch+wrist watch` | identical to `watch` | identical to `watch` | identical | 0 |

Person in view: 95.7–100 % of frames in all watch scenes.

### Keys
| Prompt | On table | In hand | No keys |
|---|---|---|---|
| `key` | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| `keys` | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| `keychain` | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| **`bunch of keys`** | **99.7 / 79.4 / 100 (0.32)**, lost events 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| `car key` | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| `key+keys+keychain` | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |

Person in view: 100 % (on table, in hand), **0 %** (no keys).

## Conclusions
- **Watch: wording is not the cause.** `watch` is the best single prompt in every scene, and alternatives are worse (worn: 92.0 % → 57.1 %). Combinations give exactly the same numbers as `watch` alone: on every frame the strongest score came from `watch`. The in-hand failure (0 % with every prompt) must come from appearance (hand covering the watch, face tilted, loose strap), not from the word.
- **Keys: wording is decisive.** `bunch of keys` raised keys on a table from 0 % to 99.7 % (no dropouts). Every single-word prompt failed. Untested hypothesis: training captions describe keys as a bunch or set.
- **Held in the hand fails for both objects with every prompt** (0 %).
- **No false sightings** with any prompt in the absent scenes. Caveat: the "no keys" scene had no person in view, so `bunch of keys` false positives with a person present are untested.
- The watch on the table reached 5.0 % here vs ~50 % in the informal live test and 0 % in EXP-06: it sits near the threshold (median confidence 0.12), so small changes in angle or light decide it.

## Live check after integration (user, 2026-10-23; informal, not measured)
`main.py --target keys --navigate --no-obstacles` with the prompt map active (`prompt: 'bunch of keys'`): the user reported that keys work in the live app.

## Caveats
One tester, one desk, one watch and one set of keys, one 10 s recording per scene. Indicative only. The prompt map proposed from this (keys → `bunch of keys`) rests on a single recording per scene.
