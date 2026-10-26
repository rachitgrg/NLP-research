from __future__ import annotations

# speech/whisper_stt.py
# ─────────────────────────────────────────────────────────────
# Module 1 – Hearing & Understanding
# WhisperTranscriber: offline speech → English with faster-whisper
# (task="translate"), the on-device replacement for Sarvam (R9).
#
# Same interface as SarvamTranscriber:
#     transcribe(audio_path) → {"text": English text, "language": code}
#
# In offline mode the model must already be in the Hugging Face
# cache (tools/offline_check.py in module2_vision lists it).
# ─────────────────────────────────────────────────────────────

import os

import numpy as np

from speech.asr_config import (
    ASR_BEAM_SIZE,
    ASR_COMPUTE_TYPE,
    ASR_DEVICE,
    ASR_LANGUAGE,
    ASR_MODEL_SIZE,
    OFFLINE_MODE,
)


WHISPER_SAMPLE_RATE = 16000


def load_audio(audio_path: str) -> np.ndarray:
    """
    Read an audio file as mono float32 at 16 kHz.

    Uses soundfile instead of faster-whisper's PyAV decoder: faster-whisper
    1.2.1 calls av.open(metadata_errors=...), which PyAV 19 removed, so its
    own decoder fails on every file in the project environment.
    """
    import soundfile as sf

    audio, rate = sf.read(audio_path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if rate != WHISPER_SAMPLE_RATE and len(audio):
        # Linear resampling; recordings from audio/microphone.py are already 16 kHz.
        n_out = int(round(len(audio) * WHISPER_SAMPLE_RATE / rate))
        audio = np.interp(np.linspace(0, len(audio) - 1, n_out), np.arange(len(audio)), audio)
    return audio.astype(np.float32)


_dll_dirs: list = []  # keep os.add_dll_directory handles alive


def register_cuda_libraries() -> bool:
    """
    Make cuBLAS 12 / cuDNN 9 loadable for CTranslate2; True if they are.

    The project's CUDA torch wheel ships both in torch/lib, but on Windows
    CTranslate2 does not look there ("cublas64_12.dll is not found").
    """
    if os.name != "nt":
        return True
    import ctypes
    from pathlib import Path

    try:
        import torch

        lib = Path(torch.__file__).parent / "lib"
        if lib.is_dir() and not _dll_dirs:
            _dll_dirs.append(os.add_dll_directory(str(lib)))
    except ImportError:
        pass
    try:
        ctypes.WinDLL("cublas64_12.dll")
        return True
    except OSError:
        return False


def default_device() -> str:
    """'cuda' if CTranslate2 sees a GPU and its CUDA libraries load, else 'cpu'."""
    try:
        import ctranslate2  # installed with faster-whisper

        if ctranslate2.get_cuda_device_count() > 0 and register_cuda_libraries():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def default_compute_type(device: str) -> str:
    """int8 weights; float16 activations on GPU."""
    return "int8_float16" if device == "cuda" else "int8"


class WhisperTranscriber:
    """
    Transcribes and translates audio to English with faster-whisper.

    Usage:
        transcriber = WhisperTranscriber(model_size="large-v3", language="hi")
        result = transcriber.transcribe("path/to/audio.wav")
        # result == {"text": "Where is my bottle?", "language": "hi"}
    """

    def __init__(
        self,
        model_size: str = ASR_MODEL_SIZE,
        device: str | None = None,
        compute_type: str | None = None,
        language: str | None = ASR_LANGUAGE,
        beam_size: int = ASR_BEAM_SIZE,
        local_files_only: bool = OFFLINE_MODE,
    ) -> None:
        """
        Load the Whisper model once.

        Args:
            model_size:   tiny / base / small / medium / large-v3, or a model path.
            device:       "cuda" or "cpu"; None = ASR_DEVICE or auto.
            compute_type: CTranslate2 compute type; None = ASR_COMPUTE_TYPE or auto.
            language:     Whisper language code of the speaker; None, "" or
                          "auto" = detect per request.
            beam_size:    Beam width for decoding.
            local_files_only: Never download (offline mode).

        Raises:
            ImportError:  If faster-whisper is not installed.
            RuntimeError: If the model is not available locally in offline mode.
        """
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise ImportError(
                "faster-whisper is required for offline speech recognition.\n"
                "Install it with:\n"
                "    pip install faster-whisper"
            ) from exc

        self.model_size = model_size
        self.device = device or ASR_DEVICE or default_device()
        if self.device == "cuda":
            register_cuda_libraries()
        self.compute_type = compute_type or ASR_COMPUTE_TYPE or default_compute_type(self.device)
        self.language = None if language in (None, "", "auto") else language
        self.beam_size = beam_size

        print(f"[Whisper] Loading '{model_size}' on {self.device} ({self.compute_type})…")
        try:
            self._model = WhisperModel(
                model_size,
                device=self.device,
                compute_type=self.compute_type,
                local_files_only=local_files_only,
            )
        except Exception as exc:
            if local_files_only:
                raise RuntimeError(
                    f"Whisper model '{model_size}' is not available offline ({exc}).\n"
                    "Download it once with OFFLINE_MODE=false, or choose a cached "
                    "model with ASR_MODEL_SIZE."
                ) from exc
            raise
        print("[Whisper] Model ready.")

    def transcribe(self, audio_path: str) -> dict:
        """
        Transcribe an audio file, translating any language to English.

        Args:
            audio_path: Path to a WAV/MP3 audio file.

        Returns:
            dict: {
                "text"     (str): English translation of the spoken audio.
                "language" (str): Whisper code of the spoken language
                                  ("hi", "en", …), as set or detected.
            }

        Raises:
            FileNotFoundError: If audio_path does not exist.
            RuntimeError:      If Whisper returns an empty transcript.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        segments, info = self._model.transcribe(
            load_audio(audio_path),
            beam_size=self.beam_size,
            task="translate",
            language=self.language,  # None → auto-detect
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=300),
        )

        # segments is a generator: decoding happens while joining.
        text = " ".join(seg.text.strip() for seg in segments).strip()
        if not text:
            raise RuntimeError(
                "Whisper returned an empty transcript. "
                "Please ensure the audio contains clear speech and try again."
            )

        return {
            "text": text,
            "language": self.language or getattr(info, "language", None) or "unknown",
        }
