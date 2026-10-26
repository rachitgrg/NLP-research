# tests/conftest.py
# ─────────────────────────────────────────────────────────────
# Module 3 – Camera & YOLO Object Detection
# pytest configuration — adds the module root to sys.path so
# that imports work correctly regardless of the CWD.
# ─────────────────────────────────────────────────────────────

import sys
from pathlib import Path

# Add module2_vision/ root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import add_module1_to_path  # noqa: E402

# TTS (audio_feedback) lives in module1_audio/
add_module1_to_path()

import socket  # noqa: E402

import pytest  # noqa: E402


@pytest.fixture()
def no_network(monkeypatch):
    """Fail the test if anything tries to open a network connection (offline mode, R9)."""
    def _blocked(*args, **kwargs):
        raise AssertionError("network access attempted in an offline test")

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)
