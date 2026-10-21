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
module1_audio/
│
├── audio/
│   └── microphone.py          # Records audio from the default microphone
│
├── audio_feedback/            # Speech output (moved here from Module 3 on 2026-10-21)
│   ├── tts_speaker.py         # Non-blocking, debounced TTS (Windows SAPI / pyttsx3)
│   └── tts_config.py          # TTS_* settings (env-var overridable)
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
│   ├── evaluate_models.py     # Benchmarks Whisper Tiny, Whisper Base, and Sarvam AI
│   ├── record_samples.py      # Records the 10 benchmark WAV files from the microphone
│   ├── metrics.py             # WER, CER, Exact Match, and accuracy computation helpers
│   ├── graph.py               # (Legacy) graph generation
│   ├── plot_results.py        # Generates the 4-panel comparison bar chart PNG
│   ├── semantic_eval.py       # Evaluates semantic extraction accuracy
│   ├── semantic_gold.csv      # Gold standard semantics for evaluation
│   ├── whisper_adapter.py     # Wraps faster-whisper with the same interface as Sarvam
│   ├── references.csv         # Expected English text for each of the 10 samples
│   ├── results.csv            # Results from the last evaluation run
│   ├── comparison_graph_v2.png # 4-panel bar chart from the last evaluation run
│   └── audio/                 # The 10 recorded WAV benchmark samples
│
├── tests/
│   ├── test_keywords.py           # Unit tests for keyword extraction
│   ├── test_query_parser.py       # Unit tests for query parsing
│   ├── test_query_parser_extended.py # Extended edge case tests for query parsing
│   ├── test_sarvam_stt.py         # Unit tests for the Sarvam STT wrapper (mocked)
│   ├── test_evaluation_metrics.py # Unit tests for WER/CER/accuracy helpers
│   ├── test_evaluation_graph.py   # Unit tests for the graph generator
│   ├── test_semantic_eval.py      # Unit tests for semantic evaluation
│   └── test_tts_speaker.py        # TTS queue, debounce and engine tests
│
├── conftest.py                # pytest config (adds module root to sys.path)
├── main.py                    # Entry point — run the interactive voice pipeline
└── README.md                  # This file
```

Dependencies are in the root `requirements.txt`; the API key goes in the root `.env`
(template: root `.env.example`). Benchmark results of record are frozen in
`../research/experiments/2026-09-16_asr_benchmark/`.

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

Use the project environment at the repository root (`.venv`). The root `README.md` → *Setup*
creates it from the root `requirements.txt` and downloads the spaCy model.

```powershell
# From the repository root: set up your Sarvam API key once
copy .env.example .env
# Open .env and paste your key from https://dashboard.sarvam.ai/

cd module1_audio
..\.venv\Scripts\Activate.ps1
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

The evaluation benchmarks **Sarvam AI** (cloud), **Whisper Tiny** (local), and **Whisper Base** (local) on 10 pre-recorded English sentences.

```bash
# From the module1_audio/ directory:

# Step 1: Record the 10 benchmark samples (only needed once)
python evaluation/record_samples.py

# Step 2: Run the benchmark
python evaluation/evaluate_models.py

# Step 3: Generate the comparison graph
python evaluation/plot_results.py

# Step 4: Evaluate semantic accuracy
python evaluation/semantic_eval.py
```

Results are saved to `evaluation/results.csv` and `evaluation/comparison_graph_v2.png`.
⚠ `evaluate_models.py` **overwrites** `evaluation/results.csv`. Before a new run, keep the old file;
afterwards, record the run as a new dated experiment in `../research/experiments/`.

---

## Running Tests

All tests use mocking — no microphone, no API key, and no audio files are needed (except the pre-recorded WAV samples in `evaluation/audio/`).
There are 96 pytest cases covering the pipeline, NLP parser, evaluation modules and TTS engine.

```powershell
# From inside module1_audio/
..\.venv\Scripts\python -m pytest tests -q
```

---

## Required Inputs

- A microphone connected to your computer
- A Sarvam AI API key (set in the project-root `.env`)
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
