# tests/conftest.py
# ─────────────────────────────────────────────────────────────
# Module 3 – Camera & YOLO Object Detection
# pytest configuration — adds the module root to sys.path so
# that imports work correctly regardless of the CWD.
# ─────────────────────────────────────────────────────────────

import sys
from pathlib import Path

# Add module-3/ root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
