# tests/test_tts_language.py
# ─────────────────────────────────────────────────────────────
# Unit tests for speech output in the user's language (roadmap R9):
# installed-voice detection and SAPI voice selection.
# No real speech engine is used.
# ─────────────────────────────────────────────────────────────

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import audio_feedback.tts_speaker as tts  # noqa: E402


@pytest.fixture(autouse=True)
def no_onecore_voices(monkeypatch):
    """Keep the real Windows OneCore voice list out of the tests."""
    monkeypatch.setattr(tts, "_onecore_tokens", lambda: [])


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


def test_select_voice_finds_onecore_voice(monkeypatch):
    # Hindi voices from Settings → Speech exist only as OneCore tokens.
    zira, kalpana = FakeToken("Zira", "409"), FakeToken("Kalpana", "439")
    monkeypatch.setattr(tts, "_onecore_tokens", lambda: [kalpana])
    engine = tts._SapiEngine.__new__(tts._SapiEngine)
    engine._voice = FakeVoice([zira])
    engine._select_voice("hi")
    assert engine._voice.Voice is kalpana


def test_select_voice_prefers_desktop_voice(monkeypatch):
    desktop, onecore = FakeToken("Zira Desktop", "409"), FakeToken("Zira", "409")
    monkeypatch.setattr(tts, "_onecore_tokens", lambda: [onecore])
    engine = tts._SapiEngine.__new__(tts._SapiEngine)
    engine._voice = FakeVoice([desktop])
    engine._select_voice("en")
    assert engine._voice.Voice is desktop


def test_sapi_voices_include_onecore(monkeypatch):
    client = pytest.importorskip("comtypes.client")
    monkeypatch.setattr(client, "CreateObject", lambda name: FakeVoice([FakeToken("Zira", "409")]))
    monkeypatch.setattr(tts.sys, "platform", "win32")
    monkeypatch.setattr(tts, "_onecore_tokens", lambda: [FakeToken("Kalpana", "439")])
    assert tts.sapi_voices() == [("Zira", ("409",)), ("Kalpana", ("439",))]


def test_speaker_records_language():
    speaker = tts.TTSSpeaker(enabled=False, use_mock=True, language="hi")
    assert speaker.language == "hi"
