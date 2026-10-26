# EXP-09 · Offline end-to-end sessions (roadmap R9, claim C19)

**Status (2026-10-27): protocol ready, not run yet.** This file is the scenario sheet to follow at the desk. Results go in the *Results* section once the sessions have run.

| | |
|---|---|
| Question | Does the full assistant (voice request → speech recognition → detection → spoken guidance) work with **no network connection**, in Hindi and English? How long does each stage take? |
| Pass criterion (C19) | Every session runs to its end with Wi-Fi off and no network error; the offline pre-flight passes. Accuracy is secondary (n is small) and is reported, not judged. |
| Script | `module2_vision/assistant.py --scenario <id>` (one search per run). The timings are logged automatically to `module2_vision/tools/results/session_latency_<date>.csv`. |
| Manual record | `observations.csv` in this folder: one row per session, filled in by hand (what was said, what was heard, whether the guidance made sense). |
| Settings | Whisper large-v3 on the GPU, language given (D-15); `OFFLINE_MODE=true` (default); obstacle warnings off for the desk (`--no-obstacles`); search timeout 60 s (`SEARCH_TIMEOUT_S`). |
| Camera | Laptop webcam (index 0). Optional repeat with a forward-facing USB camera (`--camera 1`) if one is available. |

## 1. Before you start (once)
1. Plug in the laptop charger (the GPU slows down on battery).
2. Put the objects on the desk: a **bottle**, a **cup**, your **phone**, a **bunch of keys**. Wear your **watch**.
3. Open PowerShell in the project folder and run the pre-flight while still online:
   ```powershell
   cd D:\NLP-Research\module2_vision
   ..\.venv\Scripts\python tools\offline_check.py
   ```
   It must end with **"Ready to run offline."** and list reply voices **en, hi**.
4. **Turn the network off:** switch on Airplane mode (Wi-Fi and Bluetooth off). Unplug any Ethernet cable.
5. Check that it really is off. This must **fail**:
   ```powershell
   Test-NetConnection 8.8.8.8 -Port 53 -InformationLevel Quiet
   ```
   It should print `False`. If it prints `True`, the network is still on: stop and fix that first.
6. Run the pre-flight again, now offline. It must still say "Ready to run offline."

## 2. The ten sessions
Run each row once. Wait for the "speak now" prompt; you have **5 seconds**. Say the request naturally; the wording below is only a suggestion (code-mixing such as "phone" in Hindi is fine).

**English sessions.** First set the language once in the PowerShell window:
```powershell
$env:ASR_LANGUAGE = "en"
```
Then one command per session:

| ID | Object (where) | Say | Command |
|---|---|---|---|
| E1 | bottle (on the desk) | "Where is my bottle?" | `..\.venv\Scripts\python assistant.py --no-obstacles --scenario E1_bottle` |
| E2 | phone (on the desk) | "Where is my phone?" | `..\.venv\Scripts\python assistant.py --no-obstacles --scenario E2_phone` |
| E3 | cup (on the desk) | "Find my cup." | `..\.venv\Scripts\python assistant.py --no-obstacles --scenario E3_cup` |
| E4 | watch (worn on the wrist) | "Where is my watch?" | `..\.venv\Scripts\python assistant.py --no-obstacles --scenario E4_watch` |
| E5 | keys (bunch, lying on the desk) | "Where are my keys?" | `..\.venv\Scripts\python assistant.py --no-obstacles --scenario E5_keys` |

**Hindi sessions.** Set the language once:
```powershell
$env:ASR_LANGUAGE = "hi"
```

| ID | Object (where) | Say (suggestion) | Command |
|---|---|---|---|
| H1 | bottle | "मेरी बोतल कहाँ है?" | `..\.venv\Scripts\python assistant.py --no-obstacles --scenario H1_bottle` |
| H2 | phone | "मेरा फ़ोन कहाँ है?" | `..\.venv\Scripts\python assistant.py --no-obstacles --scenario H2_phone` |
| H3 | cup | "मेरा कप ढूँढो।" | `..\.venv\Scripts\python assistant.py --no-obstacles --scenario H3_cup` |
| H4 | watch (worn) | "मेरी घड़ी कहाँ है?" | `..\.venv\Scripts\python assistant.py --no-obstacles --scenario H4_watch` |
| H5 | keys (on the desk) | "मेरी चाबी कहाँ है?" | `..\.venv\Scripts\python assistant.py --no-obstacles --scenario H5_keys` |

Why these five: bottle, phone and cup are in the fine-tuned 15-class model (closed set); watch and keys go through the open-vocabulary model (D-13, D-14). Cup was the most often misheard word in EXP-08, so it tests recognition too. A watch held in the hand and keys held in the hand are known failures (C16), so keep the watch on the wrist and the keys on the desk.

## 3. During each session
- After you speak, the terminal shows **Requested Target** and **Normalized Class**. Note what it heard.
- Follow the spoken directions: move the object (or the laptop) until it says the object is **directly in front of you**. The session then ends by itself (`arrived`).
- If the object is never found, the session ends after **60 s** (`timeout`). That's a valid result; don't restart it.
- If something goes wrong (wrong object heard, crash), press **Q** in the camera window and note it. Don't re-run silently: a second attempt gets its own row with `_retry` added to the ID (e.g. `--scenario H3_cup_retry`).
- At the end the terminal prints **Session timings** and **Logged to …**.

## 4. After each session, fill one row of `observations.csv`
| Column | What to write |
|---|---|
| `scenario` | E1_bottle … H5_keys |
| `said` | What you actually said |
| `heard_target` | The **Normalized Class** the terminal showed |
| `target_correct` | `yes` / `no` |
| `reply_language_ok` | `yes` if the replies were spoken in the language you used (Hindi voice for Hindi), else `no` |
| `guidance_ok` | `yes` if the left/right/ahead directions matched where the object really was, `no`, or `partly` |
| `finish` | `arrived` / `timeout` / `stopped (Q)` / `crash` |
| `network_off_checked` | `yes` if the step 5 check printed `False` before this session |
| `notes` | Anything odd: hesitations, repeated phrases, slow start |

## 5. When finished
1. Turn Airplane mode off.
2. Tell Claude the sessions are done. The timings CSV (`module2_vision/tools/results/session_latency_<date>.csv`) and `observations.csv` are then frozen into this folder and written up (C19, LOG, claims).

## Results
_Not run yet._

## Caveats (known in advance)
- One tester, one desk, the laptop webcam facing the user; 5 scenarios per language, one attempt each. This shows that the system *runs* offline, not how accurate it is.
- "Arrived" is the box-area heuristic (≥ 22 % of the frame, D-07), not a measured distance.
- Obstacle warnings are off (`--no-obstacles`) because the webcam faces the user at a desk.
