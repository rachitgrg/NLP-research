# tests/test_tts_language.py
# ─────────────────────────────────────────────────────────────
# Unit tests for speech output in the user's language (roadmap R9):
# installed-voice detection and SAPI voice selection.
# No real speech engine is used.
# ─────────────────────────────────────────────────────────────

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import audio_feedback.tts_speaker as tts  # noqa: E402


class FakeToken:
    def __init__(self, name, lang):
        self.name, self.lang = name, lang

    def GetDescription(self):
        return self.name

    def GetAttribute(self, key):
        return self.lang


class FakeVoice:
    def __init__(self, tokens):
        self._tokens = tokens
        self.Voice = tokens[0]

    def GetVoices(self):
        return self._tokens


def test_voice_languages_maps_windows_ids(monkeypatch):
    monkeypatch.setattr(tts, "sapi_voices", lambda: [("Zira", ("409",)), ("Kalpana", ("439",))])
    assert tts.voice_languages() == {"en", "hi"}


def test_voice_languages_always_has_english(monkeypatch):
    monkeypatch.setattr(tts, "sapi_voices", lambda: [])
    assert tts.voice_languages() == {"en"}


def test_select_voice_picks_matching_language():
    zira, kalpana = FakeToken("Zira", "409"), FakeToken("Kalpana", "439;9")
    engine = tts._SapiEngine.__new__(tts._SapiEngine)
    engine._voice = FakeVoice([zira, kalpana])
    engine._select_voice("hi")
    assert engine._voice.Voice is kalpana


def test_select_voice_keeps_default_when_missing():
    zira = FakeToken("Zira", "409")
    engine = tts._SapiEngine.__new__(tts._SapiEngine)
    engine._voice = FakeVoice([zira])
    engine._select_voice("ta")
    assert engine._voice.Voice is zira


def test_speaker_records_language():
    speaker = tts.TTSSpeaker(enabled=False, use_mock=True, language="hi")
    assert speaker.language == "hi"
