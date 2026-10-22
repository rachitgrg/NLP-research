# tests/test_detection_trials_output.py
# ─────────────────────────────────────────────────────────────
# Detection trials: each run is appended to a dated CSV together
# with the settings that produced it; earlier result files are
# never written to.
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import argparse
import csv
import time

from tools.detection_trials import save_row

METRICS = {"frames": 450, "fps": 30.0, "new_found_pct": 99.8}


def _args(scene: str = "alone plain") -> argparse.Namespace:
    return argparse.Namespace(scene=scene, target="bottle", model="weights/best.pt",
                              seconds=15.0, imgsz=640, device="cpu")


def _rows(path):
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_row_goes_to_dated_file_with_settings(tmp_path):
    path = save_row(_args(), METRICS, tmp_path)
    assert path.name == f"detection_trials_{time.strftime('%Y-%m-%d')}.csv"
    (row,) = _rows(path)
    assert row["scene"] == "alone plain" and row["model"] == "best.pt"
    for key in ("target_threshold", "base_threshold", "confirm_window", "hold_s", "imgsz", "device"):
        assert row[key] != ""


def test_same_day_runs_append_to_one_file(tmp_path):
    first = save_row(_args("a"), METRICS, tmp_path)
    second = save_row(_args("b"), METRICS, tmp_path)
    assert first == second
    assert [r["scene"] for r in _rows(first)] == ["a", "b"]


def test_existing_results_file_is_not_touched(tmp_path):
    legacy = tmp_path / "detection_trials.csv"
    legacy.write_text("timestamp,scene\n2026-10-20 19:26:22,alone plain\n", encoding="utf-8")
    before = legacy.read_bytes()
    save_row(_args(), METRICS, tmp_path)
    assert legacy.read_bytes() == before


def test_changed_columns_start_a_new_file(tmp_path):
    today = tmp_path / f"detection_trials_{time.strftime('%Y-%m-%d')}.csv"
    today.write_text("timestamp,scene,old_column\n", encoding="utf-8")
    path = save_row(_args(), METRICS, tmp_path)
    assert path != today
    assert today.read_text(encoding="utf-8") == "timestamp,scene,old_column\n"
