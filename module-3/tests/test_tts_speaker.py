# tests/test_tts_speaker.py
# ─────────────────────────────────────────────────────────────
# Module 3 – Unit tests for TTSSpeaker
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import time
import pytest

from audio_feedback.tts_speaker import SpeechPriority, TTSSpeaker


@pytest.fixture()
def mock_speaker() -> TTSSpeaker:
    speaker = TTSSpeaker(enabled=True, cooldown_s=0.5, use_mock=True)
    yield speaker
    speaker.stop()


class TestTTSSpeakerBasic:
    """Verify speech queue and callback handling."""

    def test_speak_triggers_callback(self, mock_speaker):
        received = []

        def callback(text, priority):
            received.append((text, priority))

        mock_speaker.on_speak = callback
        success = mock_speaker.speak("Testing 1 2 3", priority=SpeechPriority.GUIDANCE)
        assert success is True

        time.sleep(0.3)
        assert len(received) >= 1
        assert received[0][0] == "Testing 1 2 3"
        assert received[0][1] == SpeechPriority.GUIDANCE

    def test_disabled_speaker_drops_speech(self):
        disabled = TTSSpeaker(enabled=False, use_mock=True)
        assert disabled.speak("Hello") is False
        disabled.stop()

    def test_empty_text_dropped(self, mock_speaker):
        assert mock_speaker.speak("") is False
        assert mock_speaker.speak("   ") is False


class TestTTSSpeakerDebounce:
    """Verify cooldown and prioritization."""

    def test_repeated_text_within_cooldown_debounced(self, mock_speaker):
        assert mock_speaker.speak("Move forward", priority=SpeechPriority.GUIDANCE) is True
        # Immediately repeated -> should be debounced / rejected
        assert mock_speaker.speak("Move forward", priority=SpeechPriority.GUIDANCE) is False

    def test_emergency_priority_bypasses_cooldown(self, mock_speaker):
        assert mock_speaker.speak("Move forward", priority=SpeechPriority.GUIDANCE) is True
        # Emergency hazard warning should NOT be blocked by recent guidance
        assert mock_speaker.speak("Caution: chair ahead", priority=SpeechPriority.EMERGENCY, force=True) is True

    def test_forced_speech_bypasses_cooldown(self, mock_speaker):
        assert mock_speaker.speak("Turn right", priority=SpeechPriority.GUIDANCE) is True
        assert mock_speaker.speak("Turn right", priority=SpeechPriority.GUIDANCE, force=True) is True


class TestTTSSpeakerLifecycle:
    """Verify clean queue clearing and stopping."""

    def test_clear_queue(self, mock_speaker):
        mock_speaker.speak("Message 1")
        mock_speaker.speak("Message 2", force=True)
        mock_speaker.clear()
        assert mock_speaker._queue.empty()

    def test_stop_speaker_cleans_thread(self, mock_speaker):
        thread = mock_speaker._worker_thread
        mock_speaker.stop()
        if thread:
            assert not thread.is_alive()
        assert mock_speaker._worker_thread is None


class TestTTSSpeakerFloodProtection:
    """Per-frame repeated messages must not pile up in the queue."""

    def _idle_speaker(self):
        # Worker stopped so the queue just accumulates what speak() accepts.
        speaker = TTSSpeaker(enabled=True, cooldown_s=2.0, use_mock=True)
        speaker.stop()
        return speaker

    def test_repeated_emergency_not_requeued_within_cooldown(self):
        speaker = self._idle_speaker()
        assert speaker.speak("Caution: chair ahead", priority=SpeechPriority.EMERGENCY) is True
        for _ in range(30):
            speaker.speak("Caution: chair ahead", priority=SpeechPriority.EMERGENCY)
        assert speaker._queue.qsize() == 1

    def test_new_guidance_replaces_stale_pending_guidance(self):
        speaker = self._idle_speaker()
        speaker.speak("Turn left", priority=SpeechPriority.GUIDANCE, force=True)
        speaker.speak("Turn right", priority=SpeechPriority.GUIDANCE, force=True)
        assert speaker._queue.qsize() == 1
        assert speaker._queue.get_nowait()[2] == "Turn right"

    def test_guidance_does_not_drop_pending_emergency(self):
        speaker = self._idle_speaker()
        speaker.speak("Caution: chair ahead", priority=SpeechPriority.EMERGENCY)
        speaker.speak("Turn right", priority=SpeechPriority.GUIDANCE, force=True)
        texts = [speaker._queue.get_nowait()[2] for _ in range(speaker._queue.qsize())]
        assert texts == ["Caution: chair ahead", "Turn right"]
