# tests/test_session_log.py
# ─────────────────────────────────────────────────────────────
# Module 2 – session latency log (EXP-09) and the assistant's
# offline voice → reply-language → guidance plumbing, with the
# camera, YOLO, voice pipeline and speech engine replaced by fakes.
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import csv

import assistant
from navigation.navigation_controller import NavigationController
from navigation.session_log import FIELDS, StageTimer, append_session_row
from matching.target_matcher import TargetMatcher
from guidance.obstacle_detector import ObstacleDetector


def test_stage_timer_records_named_stages():
    timer = StageTimer()
    with timer.stage("asr_s"):
        pass
    assert set(timer.values) == {"asr_s"} and timer.values["asr_s"] >= 0


def test_append_session_row_writes_header_once(tmp_path):
    append_session_row({"target": "bottle", "asr_s": 0.4}, tmp_path)
    path = append_session_row({"target": "watch", "unknown_key": 1}, tmp_path)
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [r["target"] for r in rows] == ["bottle", "watch"]
    assert list(rows[0]) == FIELDS


class _Speaker:
    def __init__(self, *args, language="en", **kwargs):
        self.language = language
        self.said = []
        _Speaker.instances.append(self)

    def speak(self, text, priority=None, force=False):
        self.said.append(text)
        return True

    def stop(self):
        pass

    def wait_until_idle(self, timeout_s=0):
        pass


_Speaker.instances = []


def test_controller_timings():
    nav = NavigationController(
        target_name="bottle", matcher=TargetMatcher(supported_classes={0: "bottle"}),
        obstacle_detector=ObstacleDetector(enabled=False), speaker=_Speaker(), arrival_confirm_frames=1,
    )
    assert nav.timings() == {"first_sighting_s": None, "arrival_s": None}
    nav.process_frame([], 640, 480)
    big = {"class_name": "bottle", "class_id": 0, "confidence": 0.9, "bbox": [100, 50, 540, 450]}
    nav.process_frame([big], 640, 480)
    t = nav.timings()
    assert t["first_sighting_s"] is not None and t["arrival_s"] is not None


class _Detector:
    class_names = {0: "person", 1: "bottle"}
    confidence_threshold = 0.4

    def __init__(self, **kwargs):
        pass

    def set_target(self, name, conf):
        pass

    def detect(self, frame):
        return []


def test_assistant_voice_session_replies_in_hindi_and_logs(monkeypatch, tmp_path, no_network):
    _Speaker.instances = []
    rows = []
    monkeypatch.setattr(assistant, "TTSSpeaker", _Speaker)
    monkeypatch.setattr(assistant, "YOLODetector", _Detector)
    monkeypatch.setattr(assistant, "voice_languages", lambda: {"en", "hi"})
    monkeypatch.setattr(assistant, "_get_target_from_speech", lambda d, s, t: ("bottle", "hi"))
    monkeypatch.setattr(assistant, "append_session_row", lambda row: rows.append(row) or tmp_path / "x.csv")

    def fake_loop(detector, nav_controller, camera_index):
        big = {"class_name": "bottle", "class_id": 1, "confidence": 0.9, "bbox": [100, 50, 540, 450]}
        for _ in range(6):
            nav_controller.process_frame([big], 640, 480)

    monkeypatch.setattr(assistant, "run_camera_loop", fake_loop)
    assistant.run_assistant(use_mock_audio=True, obstacles=False, scenario="unit")

    hindi = _Speaker.instances[-1]
    assert hindi.language == "hi"
    assert any("ढूँढ रहे हैं" in t for t in hindi.said)                    # "looking for" in Hindi
    assert any(t.startswith("बोतल ठीक आपके सामने") for t in hindi.said)     # arrival in Hindi
    row = rows[0]
    assert row["finish_reason"] == "arrived" and row["reply_language"] == "hi"
    assert row["scenario"] == "unit" and row["detection_mode"] == "closed-set"
    assert "model_load_s" in row and row["arrival_s"] is not None
