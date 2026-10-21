# audio_feedback/tts_config.py
# ─────────────────────────────────────────────────────────────
# Module 1 – Speech output (TTS) settings.
#
# Moved here from module2_vision/config/config.py on 2026-10-21
# when TTS moved into module1_audio/. Names, defaults and env-var
# overrides are unchanged.
# ─────────────────────────────────────────────────────────────

import os

# Speech pacing and voice synthesis parameters
TTS_SPEECH_COOLDOWN_S: float = float(os.environ.get("TTS_SPEECH_COOLDOWN_S", "2.0"))
# Minimum seconds before the SAME message is spoken again (e.g. an unchanged
# "Bottle is on your left").  New messages only wait for the shorter gap
# derived from TTS_SPEECH_COOLDOWN_S.
TTS_REPEAT_COOLDOWN_S: float = float(os.environ.get("TTS_REPEAT_COOLDOWN_S", "4.0"))
# Silence spoken before each message (Windows SAPI only).  Bluetooth headsets
# clip the start of audio while their link wakes up ("ottle" for "bottle");
# the silence absorbs that.  0 disables.
TTS_LEAD_SILENCE_MS: int = int(os.environ.get("TTS_LEAD_SILENCE_MS", "250"))
TTS_SPEECH_RATE: int = int(os.environ.get("TTS_SPEECH_RATE", "220"))  # Words per minute (220 ≈ SAPI rate +2)
TTS_SPEECH_VOLUME: float = float(os.environ.get("TTS_SPEECH_VOLUME", "1.0"))  # 0.0 - 1.0
TTS_ENABLED: bool = os.environ.get("TTS_ENABLED", "true").lower() in ("true", "1", "yes")
