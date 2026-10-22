# navigation/session_log.py
# ─────────────────────────────────────────────────────────────
# Module 2 – Per-session latency log (roadmap R9 / R4, EXP-09)
#
# assistant.py appends one row per search session to
#     tools/results/session_latency_<date>.csv
# with the time of each stage (record, speech recognition, model load,
# detector setup, first sighting, arrival) and the settings used, so
# offline end-to-end runs can be reported without manual stopwatches.
# Existing rows are never rewritten.
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

from config.config import MODULE_ROOT

RESULTS_DIR = MODULE_ROOT / "tools" / "results"

FIELDS = [
    "timestamp", "scenario", "target", "detection_mode", "spoken_language", "reply_language",
    "asr_backend", "asr_model", "device", "offline_mode", "camera_index",
    "record_s", "asr_s", "model_load_s", "detector_setup_s",
    "first_sighting_s", "arrival_s", "finish_reason",
]


class StageTimer:
    """Measures named stages: ``with timer.stage("asr_s"): ...``."""

    def __init__(self) -> None:
        self.values: dict[str, float] = {}

    class _Stage:
        def __init__(self, timer: "StageTimer", name: str) -> None:
            self._timer, self._name = timer, name

        def __enter__(self) -> None:
            self._start = time.perf_counter()

        def __exit__(self, *exc: Any) -> None:
            self._timer.values[self._name] = round(time.perf_counter() - self._start, 3)

    def stage(self, name: str) -> "StageTimer._Stage":
        return StageTimer._Stage(self, name)


def append_session_row(row: dict[str, Any], results_dir: Path = RESULTS_DIR) -> Path:
    """Append ``row`` (keys from FIELDS; missing ones left empty) to today's file."""
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"session_latency_{time.strftime('%Y-%m-%d')}.csv"
    new_file = not path.exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerow({"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), **row})
    return path
