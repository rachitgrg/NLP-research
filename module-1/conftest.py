# conftest.py
# ─────────────────────────────────────────────────────────────
# pytest configuration — ensures the module-1 root is on
# sys.path so that all package imports work when pytest is
# run from any directory.
# ─────────────────────────────────────────────────────────────

import sys
import os

# Add the module-1 root (the directory containing this file) to sys.path
sys.path.insert(0, os.path.dirname(__file__))


# ── Exclude legacy/ from test collection ──────────────────────
# legacy/ contains archived Whisper code (whisper_model.py,
# test_whisper.py). faster-whisper is no longer installed, so
# those files must not be collected by pytest.
collect_ignore_glob = ["legacy/*"]
