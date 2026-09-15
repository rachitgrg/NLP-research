# speech/sarvam_stt.py
# ─────────────────────────────────────────────────────────────
# Module 1 – Hearing & Understanding
# SarvamTranscriber: wraps the Sarvam AI Saaras v3 Speech-to-Text
# API to transcribe and translate multilingual audio to English.
#
# The API key is read from the environment variable SARVAM_API_KEY.
# Never hard-code the key here. See .env.example for setup.
# ─────────────────────────────────────────────────────────────

import os

from sarvamai import SarvamAI
from sarvamai.core.api_error import ApiError


def _load_api_key() -> str:
    """
    Read the Sarvam API key from the environment.

    Raises:
        EnvironmentError: If SARVAM_API_KEY is not set.
    """
    # Support optional python-dotenv for local development.
    # Use an explicit path anchored to this file's location so it works
    # regardless of the current working directory.
    try:
        from pathlib import Path
        from dotenv import load_dotenv
        _env_path = Path(__file__).resolve().parent.parent / ".env"
        load_dotenv(dotenv_path=_env_path, override=False)
    except ImportError:
        pass  # python-dotenv not installed — rely on real env vars

    key = os.getenv("SARVAM_API_KEY", "").strip()
    if not key:
        raise EnvironmentError(
            "SARVAM_API_KEY is not set.\n"
            "Please add it to your environment or to a .env file:\n"
            "    SARVAM_API_KEY=your_sarvam_api_key_here\n"
            "Get your key at https://dashboard.sarvam.ai/"
        )
    return key


class SarvamTranscriber:
    """
    Transcribes and translates audio to English using Sarvam AI Saaras v3.

    Usage:
        transcriber = SarvamTranscriber()
        result = transcriber.transcribe("path/to/audio.wav")
        # result == {"text": "Where is my black bottle?", "language": "hi-IN"}

    The "text" field always contains the English translation.
    The "language" field contains the detected BCP-47 source language code
    (e.g. "hi-IN", "ta-IN", "bn-IN", "en-IN").
    """

    def __init__(self):
        """
        Initialise the Sarvam AI client.

        Raises:
            EnvironmentError: If SARVAM_API_KEY is missing.
        """
        api_key = _load_api_key()
        self._client = SarvamAI(api_subscription_key=api_key)

    def transcribe(self, audio_path: str) -> dict:
        """
        Send an audio file to Sarvam AI and return the English translation.

        Sarvam Saaras v3 with mode="translate" automatically detects the
        spoken language and translates speech to English in a single call.
        language_code="unknown" lets Sarvam auto-detect the source language.

        Args:
            audio_path (str): Absolute or relative path to a WAV/MP3 file.
                              Must be under 30 seconds for the synchronous API.

        Returns:
            dict: {
                "text"     (str): English translation of the spoken audio.
                "language" (str): Detected BCP-47 source language code,
                                  e.g. "hi-IN", "ta-IN", "en-IN".
                                  Falls back to "unknown" if not detected.
            }

        Raises:
            FileNotFoundError: If audio_path does not exist.
            RuntimeError:      On network failure, API error, or empty result.
        """
        # Validate the file exists before making an API call
        if not os.path.exists(audio_path):
            raise FileNotFoundError(
                f"Audio file not found: {audio_path}"
            )

        print("Sending audio to Sarvam AI...")

        try:
            with open(audio_path, "rb") as audio_file:
                response = self._client.speech_to_text.transcribe(
                    file=audio_file,
                    model="saaras:v3",
                    mode="translate",        # Translate to English
                    language_code="unknown", # Auto-detect source language
                )
        except ApiError as exc:
            # ApiError contains status_code and body but never the key
            raise RuntimeError(
                f"Sarvam AI API error (HTTP {exc.status_code}): {exc.body}"
            ) from exc
        except OSError as exc:
            raise RuntimeError(
                f"Failed to read audio file '{audio_path}': {exc}"
            ) from exc
        except Exception as exc:
            # Catch-all for network failures, timeouts, etc.
            raise RuntimeError(
                f"Sarvam AI request failed: {exc}"
            ) from exc

        # Extract the transcript (English translation)
        text = getattr(response, "transcript", None)
        if text is not None:
            text = text.strip()

        if not text:
            raise RuntimeError(
                "Sarvam AI returned an empty transcription. "
                "Please ensure the audio contains clear speech and try again."
            )

        # Extract the detected source language code
        language = getattr(response, "language_code", None) or "unknown"

        return {
            "text": text,
            "language": language,
        }
