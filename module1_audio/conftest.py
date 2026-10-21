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

