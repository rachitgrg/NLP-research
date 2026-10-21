# audio/microphone.py
# ─────────────────────────────────────────────────────────────
# Module 1 – Hearing & Understanding
# Records audio from the default system microphone and saves it
# as a temporary WAV file for Whisper to transcribe.
# ─────────────────────────────────────────────────────────────

import os
import tempfile

import numpy as np
import sounddevice as sd
import soundfile as sf


# Audio settings
SAMPLE_RATE = 16000   # 16 kHz — the rate Whisper expects
CHANNELS = 1          # Mono


def record_audio(duration: int = 5, output_path: str = None) -> str:
    """
    Record audio from the default microphone and save it as a WAV file.

    Args:
        duration (int): Recording length in seconds. Default is 5.
        output_path (str): Path where the WAV file will be saved.
                           If None, a temporary file is created automatically.

    Returns:
        str: Absolute path to the saved WAV file.

    Raises:
        RuntimeError: If no microphone is found or recording fails.
    """
    # Verify that at least one input device exists
    try:
        device_info = sd.query_devices(kind="input")
    except sd.PortAudioError as exc:
        raise RuntimeError(
            "No microphone found. Please connect a microphone and try again."
        ) from exc

    print(f"Recording for {duration} second(s)... Speak now!")

    try:
        # Record raw audio as a NumPy array: shape (samples, channels)
        audio_data = sd.rec(
            frames=int(duration * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",   # 16-bit PCM — standard for Whisper
        )
        sd.wait()  # Block until recording is finished
    except sd.PortAudioError as exc:
        raise RuntimeError(f"Microphone recording failed: {exc}") from exc

    # Decide where to save the file
    if output_path is None:
        # Create a temp file that persists until we delete it manually
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        output_path = tmp.name
        tmp.close()

    # Write the recording to disk as a WAV file
    sf.write(output_path, audio_data, SAMPLE_RATE)

    return output_path


def delete_temp_audio(audio_path: str) -> None:
    """
    Delete a temporary audio file from disk.

    Args:
        audio_path (str): Path to the file to delete.
    """
    if audio_path and os.path.exists(audio_path):
        try:
            os.remove(audio_path)
        except OSError:
            pass  # Non-fatal: temp files will be cleaned up by the OS
