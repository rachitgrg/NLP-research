# Module 1 — Hearing & Understanding

> Part of the NLP Research project.
> This module converts multilingual speech into structured English output.

---

## What This Module Does

This module listens to a voice command (in any supported language), converts it to English using the Sarvam AI cloud API, and then extracts the meaning using rule-based NLP.

```
Microphone
    ↓
Record WAV audio
    ↓
Sarvam AI Saaras v3 (cloud)
    → Auto-detects spoken language
    → Translates speech to English
    ↓
spaCy NLP (local, no internet needed)
    → Detects intent (find / identify / describe)
    → Extracts object (bottle, phone, bag…)
    → Extracts attributes (color, size, position)
    ↓
Structured JSON result
```

**Example:**
You say (in Hindi): *"मेरी काली बोतल कहाँ है?"*

```json
{
    "text": "Where is my black bottle?",
    "language": "hi-IN",
    "intent": "find_object",
    "object": "bottle",
    "attributes": { "color": "black" },
    "keywords": ["black", "bottle"]
}
```

---

## Folder Structure

```
module-1/
│
├── audio/
│   └── microphone.py          # Records audio from the default microphone
│
├── speech/
│   └── sarvam_stt.py          # Sarvam AI Saaras v3 speech-to-English API wrapper
│
├── nlp/
│   ├── keyword_extractor.py   # Extracts keywords (nouns + adjectives) using spaCy
│   └── query_parser.py        # Parses intent, object, and attributes from English text
│
├── pipeline/
│   └── hearing_pipeline.py    # Wires speech + NLP together into one call
│
├── evaluation/
│   ├── evaluate.py            # Benchmarks Whisper Tiny vs Sarvam AI on 10 WAV samples
│   ├── record_samples.py      # Records the 10 benchmark WAV files from the microphone
│   ├── metrics.py             # WER, CER, and accuracy computation helpers
│   ├── graph.py               # Generates the comparison bar chart PNG
│   ├── whisper_adapter.py     # Wraps faster-whisper with the same interface as Sarvam
│   ├── references.csv         # Expected English text for each of the 10 samples
│   ├── results.csv            # Results from the last evaluation run
│   ├── comparison_graph.png   # Bar chart from the last evaluation run
│   └── audio/                 # The 10 recorded WAV benchmark samples
│
├── tests/
│   ├── test_keywords.py           # Unit tests for keyword extraction
│   ├── test_query_parser.py       # Unit tests for query parsing
│   ├── test_sarvam_stt.py         # Unit tests for the Sarvam STT wrapper (mocked)
│   ├── test_evaluation_metrics.py # Unit tests for WER/CER/accuracy helpers
│   └── test_evaluation_graph.py   # Unit tests for the graph generator
│
├── conftest.py                # pytest config (adds module root to sys.path)
├── main.py                    # Entry point — run the interactive voice pipeline
├── requirements.txt           # Python dependencies with install instructions
├── .env.example               # Template for the required Sarvam API key
└── README.md                  # This file
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `sarvamai` | Sarvam AI Python SDK (Saaras v3 cloud STT) |
| `python-dotenv` | Load `SARVAM_API_KEY` from `.env` file |
| `spacy` | Rule-based NLP for keyword/intent extraction |
| `sounddevice` | Microphone recording |
| `soundfile` | Read/write WAV files |
| `numpy` | Audio buffer handling |
| `pytest` | Running unit tests |
| `faster-whisper` | Evaluation only — local Whisper Tiny model |
| `jiwer` | WER/CER computation for evaluation |
| `matplotlib` | Generates the evaluation comparison chart |

---

## Setup

### Step 1 — Create and activate a virtual environment

```powershell
# Windows (PowerShell) — run from inside module-1/
python -m venv venv
.\venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Download the spaCy English model (run once)

```bash
python -m spacy download en_core_web_sm
```

### Step 4 — Set up your Sarvam API key

```bash
# Copy the template and edit it
cp .env.example .env
# Open .env and paste your key from https://dashboard.sarvam.ai/
```

---

## Running the Pipeline

```bash
# Default: record for 5 seconds
python main.py

# Custom recording duration
python main.py --duration 8
```

The pipeline will:
1. Ask you to press **Enter** to start recording
2. Record your voice for the specified duration
3. Send the audio to Sarvam AI for transcription
4. Parse the English result with spaCy
5. Print the structured JSON output

Press `q` then **Enter** to quit.

---

## Running the Evaluation

The evaluation benchmarks **Whisper Tiny** (local, CPU) vs **Sarvam AI** (cloud) on 10 pre-recorded English sentences.

```bash
# From the module-1/ directory:

# Step 1: Record the 10 benchmark samples (only needed once)
python evaluation/record_samples.py

# Step 2: Run the benchmark
python evaluation/evaluate.py
```

Results are saved to `evaluation/results.csv` and `evaluation/comparison_graph.png`.

---

## Running Tests

All tests use mocking — no microphone, no API key, and no audio files are needed (except the pre-recorded WAV samples in `evaluation/audio/`).

```bash
# Run all tests from inside module-1/
pytest tests/ -v
```

---

## Required Inputs

- A microphone connected to your computer
- A Sarvam AI API key (set in `.env`)
- Internet connection (for the Sarvam AI cloud API)

---

## Expected Output

The pipeline returns a Python `dict` (also printed as JSON):

| Key | Type | Description |
|---|---|---|
| `text` | `str` | English translation of what was said |
| `language` | `str` | Detected source language (BCP-47, e.g. `hi-IN`) |
| `intent` | `str` | `find_object`, `describe_object`, `identify_object`, `general_question`, or `unknown` |
| `object` | `str\|None` | The primary noun (e.g. `bottle`, `phone`) |
| `attributes` | `dict` | Extracted attributes: `color`, `size`, `position` |
| `keywords` | `list[str]` | All meaningful keywords in order |

---

## Important Notes

- The Sarvam AI API requires an **internet connection** and a valid **API key**.
- The spaCy NLP step is **fully local** — no API or internet is needed after setup.
- Supported source languages include Hindi, Tamil, Bengali, Kannada, Telugu, Malayalam, and more. See the [Sarvam AI documentation](https://docs.sarvam.ai/) for the full list.
- Audio must be **under 30 seconds** for the synchronous API.

---

## Limitations

- Intent detection is rule-based and limited to the four categories listed above.
- Attribute extraction only covers colours, sizes, and positions defined in `nlp/query_parser.py`.
- Recognition accuracy depends on audio quality (microphone, background noise, speaking clarity).
- The Sarvam AI API may introduce latency depending on network conditions.
