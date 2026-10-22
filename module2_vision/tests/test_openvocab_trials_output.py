# tests/test_openvocab_trials_output.py
# ─────────────────────────────────────────────────────────────
# Module 2 – result files written by tools/openvocab_trials.py
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import csv

from tools.openvocab_trials import save_rows


def _rows(**extra):
    return [{"scene": "s", "target": "watch", "model": "m", "new_found_pct": 50.0, **extra}]


def test_rows_appended_to_dated_file(tmp_path):
    p1 = save_rows(_rows(), tmp_path)
    p2 = save_rows(_rows(), tmp_path)
    assert p1 == p2 and p1.name.startswith("openvocab_trials_")
    with p1.open(encoding="utf-8") as fh:
        assert len(list(csv.DictReader(fh))) == 2


def test_changed_columns_go_to_a_new_file(tmp_path):
    p1 = save_rows(_rows(), tmp_path)
    p2 = save_rows(_rows(extra_col=1), tmp_path)
    assert p1 != p2
    with p1.open(encoding="utf-8") as fh:
        assert "extra_col" not in next(csv.reader(fh))


def test_parse_prompt_sets():
    from tools.openvocab_trials import parse_prompt_sets
    assert parse_prompt_sets(None, "watch") == [["watch"]]
    assert parse_prompt_sets(["watch", "Wristwatch", "watch+wristwatch"], "watch") == [
        ["watch"], ["wristwatch"], ["watch", "wristwatch"]]


def test_later_runs_reuse_the_matching_timestamped_file(tmp_path):
    """Regression: every run with new columns used to open yet another file."""
    old = save_rows(_rows(), tmp_path)                 # today's file, old columns
    p1 = save_rows(_rows(prompts="watch"), tmp_path)   # new columns → timestamped file
    p2 = save_rows(_rows(prompts="key"), tmp_path)     # same new columns → same file
    assert p1 != old and p1 == p2
    with p1.open(encoding="utf-8") as fh:
        assert len(list(csv.DictReader(fh))) == 2
