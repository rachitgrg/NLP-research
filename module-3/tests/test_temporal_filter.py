# tests/test_temporal_filter.py
# ─────────────────────────────────────────────────────────────
# Module 3 – TargetTracker (temporal smoothing) and the
# detection-trials analysis that compares OLD vs NEW pipelines.
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

from audio_feedback.tts_speaker import TTSSpeaker
from matching.temporal_filter import TargetTracker
from navigation.navigation_controller import NavigationController
from tools.detection_trials import Sample, analyze

BOX = [100, 100, 160, 220]


def _match(found: bool, bbox=BOX):
    found = bool(found)
    return {"target": "bottle", "found": found, "confidence": 0.5 if found else None,
            "bbox": bbox if found else None, "matched_class": "bottle" if found else None,
            "reason": None}


def _run(tracker, pattern, fps=20.0):
    return [tracker.update(_match(f), i / fps)["found"] for i, f in enumerate(pattern)]


class TestTargetTracker:
    def test_single_frame_false_positive_is_ignored(self):
        assert _run(TargetTracker(5, 2, 0.5), [0, 0, 1, 0, 0, 0, 0, 0]) == [False] * 8

    def test_confirmed_after_two_hits_in_window(self):
        out = _run(TargetTracker(5, 2, 0.5), [1, 0, 1, 1])
        assert out == [False, False, True, True]

    def test_short_gap_is_held_with_last_box(self):
        tr = TargetTracker(5, 2, 0.5)
        _run(tr, [1, 1])
        held = tr.update(_match(False), 0.2)       # 0.15 s after last sighting
        assert held["found"] is True and held["bbox"] == BOX

    def test_long_gap_reports_lost_and_needs_reconfirmation(self):
        tr = TargetTracker(5, 2, 0.5)
        _run(tr, [1, 1])
        assert tr.update(_match(False), 1.0)["found"] is False   # 0.95 s gap
        for t in (1.05, 1.10, 1.15, 1.20, 1.25):
            tr.update(_match(False), t)                         # window now empty
        assert tr.update(_match(True), 1.30)["found"] is False  # one hit is not enough
        assert tr.update(_match(True), 1.35)["found"] is True

    def test_flicker_at_70_percent_stays_found(self):
        pattern = [1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0] * 3
        out = _run(TargetTracker(5, 2, 0.5), pattern)
        assert all(out[1:])

    def test_unsupported_passes_through(self):
        r = {"target": "watch", "found": False, "confidence": None, "bbox": None,
             "matched_class": None, "reason": "unsupported_class"}
        assert TargetTracker().update(r, 0.0) is r

    def test_controller_uses_tracker_and_does_not_report_lost_on_one_miss(self):
        nc = NavigationController("bottle", speaker=TTSSpeaker(enabled=True, use_mock=True),
                                  tracker=TargetTracker(5, 2, 0.5))
        bottle = [{"class_name": "bottle", "class_id": 5, "confidence": 0.6, "bbox": BOX}]
        assert nc.process_frame(bottle, 640, 480).target_found is False   # 1st hit: not yet confirmed
        assert nc.process_frame(bottle, 640, 480).target_found is True
        assert nc.process_frame([], 640, 480).target_found is True        # held
        assert nc.state.value == "GUIDING"


class TestDetectionTrialsAnalysis:
    def _samples(self, confs, fps=20.0):
        return [Sample(t=i / fps, target_conf=c, bbox=BOX if c > 0 else None, person=False)
                for i, c in enumerate(confs)]

    def test_old_vs_new_from_same_frames(self):
        # alternating 0.30 / 0.45: OLD sees only half, NEW sees (almost) all
        m = analyze(self._samples([0.30, 0.45] * 20), base_conf=0.40, target_conf=0.25,
                    tracker=TargetTracker(5, 2, 0.5))
        assert m["old_found_pct"] == 50.0
        assert m["new_found_pct"] >= 95.0
        assert m["old_lost_events"] > m["new_lost_events"]

    def test_no_detections(self):
        m = analyze(self._samples([0.0] * 30))
        assert m["old_found_pct"] == 0.0 and m["new_found_pct"] == 0.0
        assert m["median_conf"] == 0.0
        assert m["old_longest_gap_s"] > 1.0
