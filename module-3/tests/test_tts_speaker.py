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


class FakeEngine:
    """Stands in for the SAPI/pyttsx3 backend; records what was spoken."""

    def __init__(self, duration_s: float = 0.2):
        self.spoken: list[str] = []
        self.duration_s = duration_s
        self.closed = False

    def speak(self, text):
        time.sleep(self.duration_s)
        self.spoken.append(text)

    def close(self):
        self.closed = True


def _speaker_with_engine(monkeypatch, engine, **kwargs) -> TTSSpeaker:
    monkeypatch.setattr(TTSSpeaker, "_init_engine", lambda self: engine)
    return TTSSpeaker(enabled=True, **kwargs)


class TestTTSSpeakerEngineDelivery:
    """Every accepted message must reach the engine — not just the first."""

    def test_successive_direction_changes_all_reach_engine(self, monkeypatch):
        engine = FakeEngine(duration_s=0.05)
        sp = _speaker_with_engine(monkeypatch, engine, cooldown_s=0.0, repeat_cooldown_s=0.0)
        for text in ["Bottle is on your left.", "Bottle is straight ahead.", "Bottle is on your right."]:
            assert sp.speak(text) is True
            assert sp.wait_until_idle(timeout_s=2.0)
        sp.stop()
        assert engine.spoken == [
            "Bottle is on your left.", "Bottle is straight ahead.", "Bottle is on your right."
        ]
        assert engine.closed

    def test_silent_engine_is_reported(self, monkeypatch, caplog):
        engine = FakeEngine(duration_s=0.0)  # returns instantly = no audible speech
        sp = _speaker_with_engine(monkeypatch, engine)
        with caplog.at_level("WARNING", logger="audio_feedback.tts_speaker"):
            sp.speak("Bottle is on your left.", force=True)
            sp.wait_until_idle(timeout_s=2.0)
        sp.stop()
        assert "probably not heard" in caplog.text

    def test_engine_init_failure_falls_back_and_logs(self, monkeypatch, caplog):
        import audio_feedback.tts_speaker as mod

        class Broken:
            def __init__(self, *a):
                raise OSError("no audio device")

        monkeypatch.setattr(mod, "_SapiEngine", Broken)
        monkeypatch.setattr(mod, "_Pyttsx3Engine", Broken)
        with caplog.at_level("ERROR", logger="audio_feedback.tts_speaker"):
            sp = TTSSpeaker(enabled=True)
            assert sp.speak("Hello there", force=True) is True
            assert sp.wait_until_idle(timeout_s=2.0)
            sp.stop()
        assert "Speech output unavailable" in caplog.text


class TestTTSSpeakerRepeatCadence:
    """Unchanged guidance repeats only after the repeat cooldown (A3)."""

    def test_identical_message_waits_for_repeat_cooldown(self, monkeypatch):
        import audio_feedback.tts_speaker as mod
        clock = [100.0]
        monkeypatch.setattr(mod.time, "monotonic", lambda: clock[0])
        sp = TTSSpeaker(enabled=True, use_mock=True, cooldown_s=2.0, repeat_cooldown_s=4.0)
        sp.stop()
        assert sp.speak("Bottle is on your left.") is True
        clock[0] += 2.5
        assert sp.speak("Bottle is on your left.") is False   # same text, < 4 s
        clock[0] += 2.0
        assert sp.speak("Bottle is on your left.") is True    # 4.5 s later

    def test_new_direction_only_waits_short_gap(self, monkeypatch):
        import audio_feedback.tts_speaker as mod
        clock = [100.0]
        monkeypatch.setattr(mod.time, "monotonic", lambda: clock[0])
        sp = TTSSpeaker(enabled=True, use_mock=True, cooldown_s=2.0, repeat_cooldown_s=4.0)
        sp.stop()
        assert sp.speak("Bottle is on your left.") is True
        clock[0] += 1.3   # > 0.6 * cooldown_s
        assert sp.speak("Bottle is straight ahead.") is True
