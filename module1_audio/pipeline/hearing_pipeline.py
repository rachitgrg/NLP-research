# pipeline/hearing_pipeline.py
# ─────────────────────────────────────────────────────────────
# Module 1 – Hearing & Understanding
# Orchestrates the full pipeline:
#
#   audio file
#       ↓
#   Speech recogniser    →  English text  (+ detected language)
#   (offline Whisper by default; Sarvam opt-in — speech/transcriber_factory.py)
#       ↓
#   Query Parser          →  intent, object, attributes, keywords
#       ↓
#   Structured dict  (JSON-compatible)
# ─────────────────────────────────────────────────────────────

from speech.transcriber_factory import create_transcriber
from nlp.keyword_extractor import load_nlp_model
from nlp.query_parser import parse_query


class HearingPipeline:
    """
    Loads models once and exposes a single `run()` method.

    Usage:
        pipeline = HearingPipeline()
        result   = pipeline.run("path/to/audio.wav")
        print(result)
    """

    def __init__(self, nlp_model_name: str = "en_core_web_sm", transcriber=None):
        """
        Initialise and load all models (done once at startup).

        Args:
            nlp_model_name (str): spaCy model name.
            transcriber: Object with transcribe(path) -> {"text", "language"};
                         default: create_transcriber() (ASR_BACKEND).
        """
        self.transcriber = transcriber if transcriber is not None else create_transcriber()
        self.nlp_model   = load_nlp_model(nlp_model_name)

    def run(self, audio_path: str) -> dict:
        """
        Transcribe an audio file and parse the query for intent, object and attributes.

        The pipeline is:
            audio_path
                → transcriber        (speech → English, e.g. Whisper translate)
                → English text
                → parse_query (spaCy NLP)
                → structured result

        Args:
            audio_path (str): Path to the WAV file to process.

        Returns:
            dict: {
                "text"       (str):        English translation of the spoken audio.
                "language"   (str):        Source language code ("hi", "hi-IN", …).
                "intent"     (str):        The user intent.
                "object"     (str|None):   The primary object noun.
                "attributes" (dict):       Extracted attributes (color, size, etc).
                "keywords"   (list[str]):  All extracted keywords.
            }
        """
        # Step 1 — Transcribe & translate to English
        transcript = self.transcriber.transcribe(audio_path)

        english_text      = transcript["text"]
        detected_language = transcript["language"]

        print(f"\nDetected language:\n{detected_language}")
        print(f"\nTranslated text:\n{english_text}")

        # Step 2 — Parse Query (Intent, Object, Attributes, Keywords)
        # parse_query receives plain English text — exactly as before
        parsed_result = parse_query(self.nlp_model, english_text)

        # Overwrite the language field with the recogniser's source language
        parsed_result["language"] = detected_language

        return parsed_result
