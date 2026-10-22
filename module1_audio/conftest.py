# conftest.py
# ─────────────────────────────────────────────────────────────
# pytest configuration — ensures the module1_audio root is on
# sys.path so that all package imports work when pytest is
# run from any directory.
# ─────────────────────────────────────────────────────────────

import sys
import os

# Add the module1_audio root (the directory containing this file) to sys.path
sys.path.insert(0, os.path.dirname(__file__))


import socket

import pytest


@pytest.fixture()
def no_network(monkeypatch):
    """Fail the test if anything tries to open a network connection (offline mode, R9)."""
    def _blocked(*args, **kwargs):
        raise AssertionError("network access attempted in an offline test")

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)
