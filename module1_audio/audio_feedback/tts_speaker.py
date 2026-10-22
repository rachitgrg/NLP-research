# audio_feedback/tts_speaker.py
# ─────────────────────────────────────────────────────────────
# Module 1 – Audio Feedback & Speech Synthesis
# (moved from Module 3 / Milestone 5 on 2026-10-21)
#
# Thread-safe, non-blocking Text-To-Speech engine.
# Designed for blind and visually impaired users:
#   • Runs audio synthesis in a background worker thread.
#   • Enforces utterance cooldown / debouncing to avoid speech flood.
#   • Supports priority interruptions (obstacles & arrival).
#   • Gracefully degrades to a mock console logger if audio hardware
#     is unavailable or in automated test environments.
#
# Speech backends
# ───────────────
#   Windows : SAPI5 called directly via comtypes (blocking Speak()).
#   Other   : pyttsx3, with a fresh engine per utterance.
#
# A single reused pyttsx3 engine only speaks the FIRST utterance on
# Windows when driven from a worker thread: later runAndWait() calls
# return in ~0.07 s without sound or error.  Both backends above avoid
# reusing pyttsx3's event-loop state.
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import logging
import queue
import sys
import threading
import time
from enum import IntEnum
from typing import Any, Callable, Optional
from xml.sax.saxutils import escape as xml_escape

from audio_feedback.tts_config import (
    TTS_ENABLED,
    TTS_LEAD_SILENCE_MS,
    TTS_REPEAT_COOLDOWN_S,
    TTS_SPEECH_COOLDOWN_S,
    TTS_SPEECH_RATE,
    TTS_SPEECH_VOLUME,
)

logger = logging.getLogger(__name__)

# An utterance of real text that "finishes" faster than this almost
# certainly produced no sound (the silent-failure signature above).
_SILENT_SPEECH_THRESHOLD_S = 0.15

# Windows language IDs (hex, as SAPI voice tokens report them) per
# language code, for replies in the user's language (roadmap R9).
SAPI_LANGUAGE_IDS: dict[str, tuple[str, ...]] = {
    "en": ("409", "809", "4009", "c09", "1009"),
    "hi": ("439",), "bn": ("445",), "pa": ("446",), "gu": ("447",), "ta": ("449",),
    "te": ("44a",), "kn": ("44b",), "ml": ("44c",), "mr": ("44e",), "ur": ("420",),
}


def sapi_voices() -> list[tuple[str, tuple[str, ...]]]:
    """Installed SAPI voices as (description, language IDs); [] off Windows."""
    if sys.platform != "win32":
        return []
    try:
        import comtypes.client  # type: ignore

        tokens = comtypes.client.CreateObject("SAPI.SpVoice").GetVoices()
        return [
            (t.GetDescription(), tuple(x.strip().lower() for x in t.GetAttribute("Language").split(";")))
            for t in tokens
        ]
    except Exception as exc:
        logger.warning("Could not list SAPI voices: %s", exc)
        return []


def voice_languages() -> set[str]:
    """Language codes with an installed voice (always includes "en")."""
    found = {"en"}
    for _, ids in sapi_voices():
        found |= {code for code, codes in SAPI_LANGUAGE_IDS.items() if set(ids) & set(codes)}
    return found


class _SapiEngine:
    """Windows SAPI5 voice; must be created and used in the same thread."""

    _SVSF_IS_XML = 8  # SpeechVoiceSpeakFlags: text contains SAPI XML

    def __init__(self, rate_wpm: int, volume: float, lead_silence_ms: int = 0, language: str = "en") -> None:
        self._lead_silence_ms = max(0, lead_silence_ms)
        import comtypes  # type: ignore
        import comtypes.client  # type: ignore

        comtypes.CoInitialize()
        self._comtypes = comtypes
        self._voice = comtypes.client.CreateObject("SAPI.SpVoice")
        # SAPI rate is -10..10 (0 ≈ 180 wpm); volume is 0..100.
        self._voice.Rate = max(-10, min(10, round((rate_wpm - 180) / 18)))
        self._voice.Volume = max(0, min(100, int(volume * 100)))
        if language != "en":
            self._select_voice(language)

    def _select_voice(self, language: str) -> None:
        """Use the first installed voice for ``language`` (default voice if none)."""
        wanted = set(SAPI_LANGUAGE_IDS.get(language, ()))
        for token in self._voice.GetVoices():
            ids = {x.strip().lower() for x in token.GetAttribute("Language").split(";")}
            if ids & wanted:
                self._voice.Voice = token
                return
        logger.warning("No SAPI voice for language %r; using the default voice.", language)

    def speak(self, text: str) -> None:
        # Synchronous (no async flag): returns after speaking.
        self._voice.Speak(*self.prepare(text, self._lead_silence_ms))

    @classmethod
    def prepare(cls, text: str, lead_silence_ms: int) -> tuple[str, int]:
        """
        Return (text, flags) for SpVoice.Speak.

        Bluetooth headsets and some sound cards wake their audio link only
        when sound starts, clipping the first ~100-300 ms ("ottle" for
        "bottle").  A leading SAPI <silence/> absorbs that wake-up time.
        """
        if lead_silence_ms <= 0:
            return text, 0
        return f'<silence msec="{lead_silence_ms}"/>{xml_escape(text)}', cls._SVSF_IS_XML

    def close(self) -> None:
        self._voice = None
        self._comtypes.CoUninitialize()


class _Pyttsx3Engine:
    """pyttsx3 fallback (non-Windows); fresh engine per utterance."""

    def __init__(self, rate_wpm: int, volume: float) -> None:
        import pyttsx3  # type: ignore

        self._pyttsx3 = pyttsx3
        self._rate = rate_wpm
        self._volume = volume
        self._make().stop()  # fail early if no backend is available

    def _make(self) -> Any:
        engine = self._pyttsx3.init()
        engine.setProperty("rate", self._rate)
        engine.setProperty("volume", self._volume)
        return engine

    def speak(self, text: str) -> None:
        engine = self._make()
        try:
            engine.say(text)
            engine.runAndWait()
        finally:
            engine.stop()
            del engine

    def close(self) -> None:
        pass


class SpeechPriority(IntEnum):
    """
    Priority levels for spoken navigation cues.
    Lower numerical value = higher priority.
    """
    EMERGENCY = 1   # Obstacle collision warning / immediate hazard
    ARRIVAL   = 2   # Target reached
    GUIDANCE  = 3   # Turn left / right / move straight
    STATUS    = 4   # Scanning / searching status


class TTSSpeaker:
    """
    Non-blocking Text-to-Speech manager.
    """

    def __init__(
        self,
        enabled: bool = TTS_ENABLED,
        cooldown_s: float = TTS_SPEECH_COOLDOWN_S,
        speech_rate: int = TTS_SPEECH_RATE,
        speech_volume: float = TTS_SPEECH_VOLUME,
        use_mock: bool = False,
        repeat_cooldown_s: float = TTS_REPEAT_COOLDOWN_S,
        language: str = "en",
    ) -> None:
        # Voice language (code such as "en", "hi"); see voice_languages().
        self.language = language
        self.enabled = enabled
        self.cooldown_s = cooldown_s
        self.repeat_cooldown_s = repeat_cooldown_s
        self._silent_count = 0
        self.speech_rate = speech_rate
        self.speech_volume = speech_volume
        self.use_mock = use_mock

        # Internal queue: (priority, timestamp, text)
        self._queue: queue.PriorityQueue[tuple[int, float, str]] = queue.PriorityQueue()
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        # History tracking for debouncing
        self._last_spoken_text: str = ""
        self._last_spoken_time: float = 0.0

        # Optional callback for test verification
        self.on_speak: Optional[Callable[[str, SpeechPriority], None]] = None

        # Background worker thread
        self._worker_thread: Optional[threading.Thread] = None
        if self.enabled:
            self._start_worker()

    def _start_worker(self) -> None:
        """Start background TTS worker."""
        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="TTSSpeakerWorker",
        )
        self._worker_thread.start()

    def speak(
        self,
        text: str,
        priority: SpeechPriority = SpeechPriority.GUIDANCE,
        force: bool = False,
    ) -> bool:
        """
        Request spoken output for the given text.

        Parameters
        ----------
        text : str
            The text to speak aloud.
        priority : SpeechPriority
            Priority of the message.
        force : bool
            If True, ignores the debounce timer (used for emergencies).

        Returns
        -------
        bool
            True if queued successfully, False if dropped due to debouncing or disabled.
        """
        if not self.enabled or not text or not text.strip():
            return False

        cleaned = text.strip()
        now = time.monotonic()

        with self._lock:
            if not force:
                # Never repeat the same text within the repeat cooldown — this
                # applies to EMERGENCY/ARRIVAL too, which are re-issued on every
                # frame (~20 FPS) while the condition persists.
                repeat_window = max(self.cooldown_s, self.repeat_cooldown_s)
                if cleaned == self._last_spoken_text and (now - self._last_spoken_time) < repeat_window:
                    return False
                if priority >= SpeechPriority.GUIDANCE and (now - self._last_spoken_time) < (self.cooldown_s * 0.6):
                    # Minimum pause between distinct regular guidance instructions
                    return False

            self._last_spoken_text = cleaned
            self._last_spoken_time = now

            # Pending messages of equal or lower importance are now stale
            # (e.g. an old "turn left" once a newer instruction exists).
            self._drop_pending(min_priority=int(priority))

            self._queue.put((int(priority), now, cleaned))
        return True

    def _drop_pending(self, min_priority: int) -> None:
        """Remove queued items whose priority value is >= min_priority."""
        kept: list[tuple[int, float, str]] = []
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            self._queue.task_done()
            if item[0] < min_priority:
                kept.append(item)
        for item in kept:
            self._queue.put(item)

    def _init_engine(self) -> Any:
        """
        Create the speech backend in the worker thread.

        Returns an object with ``speak(text)`` (blocks until spoken) and
        ``close()``, or None for console-only output.
        """
        if self.use_mock:
            return None

        try:
            if sys.platform == "win32":
                engine_cls = _SapiEngine
                return _SapiEngine(self.speech_rate, self.speech_volume, TTS_LEAD_SILENCE_MS, self.language)
            engine_cls = _Pyttsx3Engine
            return _Pyttsx3Engine(self.speech_rate, self.speech_volume)
        except Exception as exc:
            logger.error(
                "Speech output unavailable (%s: %s). Guidance will be printed only.",
                engine_cls.__name__, exc,
            )
            return None

    def _speak_with_engine(self, engine: Any, text: str) -> None:
        """Speak one utterance and flag the silent-failure signature."""
        t0 = time.monotonic()
        engine.speak(text)
        elapsed = time.monotonic() - t0
        logger.debug("Spoke %r in %.2fs", text, elapsed)

        if elapsed < _SILENT_SPEECH_THRESHOLD_S and len(text) >= 3:
            self._silent_count += 1
            # First occurrence and then every 10th, to avoid console spam.
            if self._silent_count == 1 or self._silent_count % 10 == 0:
                logger.warning(
                    "TTS returned after %.2fs for %r — speech was probably not heard "
                    "(%d such utterance(s) so far). Check the audio output device.",
                    elapsed, text, self._silent_count,
                )

    def _worker_loop(self) -> None:
        """Continuously pops from priority queue and synthesizes speech."""
        engine = self._init_engine()

        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            priority_int, _, text = item
            priority = SpeechPriority(priority_int)

            try:
                if self.on_speak:
                    self.on_speak(text, priority)

                if engine is not None:
                    self._speak_with_engine(engine, text)
                else:
                    # Mock or console mode
                    # ASCII only: a cp1252 Windows console cannot encode emoji,
                    # and the resulting error would hide the guidance text.
                    print(f"  [AUDIO] {text}")
            except Exception as exc:
                logger.error("TTS speech error for %r: %s", text, exc)
            finally:
                self._queue.task_done()

        if engine is not None:
            try:
                engine.close()
            except Exception:
                pass

    def wait_until_idle(self, timeout_s: float = 5.0) -> bool:
        """Block until queued speech has been spoken (or timeout). Returns True if idle."""
        deadline = time.monotonic() + timeout_s
        while self._queue.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(0.05)
        return not self._queue.unfinished_tasks

    def stop(self) -> None:
        """Stop worker thread and clear queue."""
        self._stop_event.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
        self._worker_thread = None

    def clear(self) -> None:
        """Clear all pending utterances."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                break
