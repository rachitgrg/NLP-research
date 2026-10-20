# audio_feedback/tts_speaker.py
# ─────────────────────────────────────────────────────────────
# Module 3 – Milestone 5: Audio Feedback & Speech Synthesis
#
# Thread-safe, non-blocking Text-To-Speech engine using pyttsx3.
# Designed for blind and visually impaired users:
#   • Runs audio synthesis in a background worker thread.
#   • Enforces utterance cooldown / debouncing to avoid speech flood.
#   • Supports priority interruptions (obstacles & arrival).
#   • Gracefully degrades to a mock console logger if audio hardware
#     is unavailable or in automated test environments.
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import logging
import queue
import threading
import time
from enum import IntEnum
from typing import Callable, Optional

from config.config import (
    TTS_ENABLED,
    TTS_SPEECH_COOLDOWN_S,
    TTS_SPEECH_RATE,
    TTS_SPEECH_VOLUME,
)

logger = logging.getLogger(__name__)


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
    ) -> None:
        self.enabled = enabled
        self.cooldown_s = cooldown_s
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
                # Never repeat the same text within the cooldown — this applies
                # to EMERGENCY/ARRIVAL too, which are re-issued on every frame
                # (~20 FPS) while the condition persists.
                if cleaned == self._last_spoken_text and (now - self._last_spoken_time) < self.cooldown_s:
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
        """Initialise the pyttsx3 engine in the worker thread."""
        if self.use_mock:
            return None

        try:
            import pyttsx3  # type: ignore
            engine = pyttsx3.init()
            engine.setProperty("rate", self.speech_rate)
            engine.setProperty("volume", self.speech_volume)
            return engine
        except Exception as exc:
            logger.warning("Could not initialize pyttsx3 (%s). Falling back to mock audio logger.", exc)
            return None

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
                    engine.say(text)
                    engine.runAndWait()
                else:
                    # Mock or console mode
                    print(f"  🔊 [AUDIO] {text}")
            except Exception as exc:
                logger.error("TTS speech error: %s", exc)
            finally:
                self._queue.task_done()

        if engine is not None:
            try:
                engine.stop()
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
