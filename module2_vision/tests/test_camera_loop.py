# tests/test_camera_loop.py
# ─────────────────────────────────────────────────────────────
# Module 3 – camera loop session control
# Runs run_camera_loop() with a fake camera, detector and OpenCV
# GUI so loop termination can be tested without hardware.
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import numpy as np
import pytest

import detection.camera_detector as cam
from audio_feedback.tts_speaker import TTSSpeaker
from navigation.navigation_controller import NavigationController

_BIG_BOTTLE = [{"class_name": "bottle", "class_id": 5, "confidence": 0.9, "bbox": [100, 50, 540, 450]}]


class FakeCapture:
    def __init__(self, frames_ok: bool = True, max_reads: int = 500):
        self.frames_ok = frames_ok
        self.reads = 0
        self.max_reads = max_reads
        self.released = False

    def isOpened(self):
        return True

    def read(self):
        self.reads += 1
        if self.reads > self.max_reads:
            raise AssertionError("loop did not terminate")
        if not self.frames_ok:
            return False, None
        return True, np.zeros((480, 640, 3), dtype=np.uint8)

    def release(self):
        self.released = True


class FakeDetector:
    confidence_threshold = 0.4

    def __init__(self, detections):
        self._detections = detections

    def detect(self, frame):
        return list(self._detections)


@pytest.fixture()
def fake_gui(monkeypatch):
    monkeypatch.setattr(cam.cv2, "namedWindow", lambda *a, **k: None)
    monkeypatch.setattr(cam.cv2, "imshow", lambda *a, **k: None)
    monkeypatch.setattr(cam.cv2, "waitKey", lambda *a, **k: -1)
    monkeypatch.setattr(cam.cv2, "getWindowProperty", lambda *a, **k: 1.0)
    monkeypatch.setattr(cam.cv2, "destroyAllWindows", lambda *a, **k: None)
    monkeypatch.setattr(cam.time, "sleep", lambda s: None)


def test_loop_ends_when_target_reached(monkeypatch, fake_gui):
    capture = FakeCapture()
    monkeypatch.setattr(cam, "_open_camera", lambda idx, backend: capture)
    speaker = TTSSpeaker(enabled=False)
    nav = NavigationController("bottle", speaker=speaker, arrival_confirm_frames=3)

    cam.run_camera_loop(FakeDetector(_BIG_BOTTLE), nav_controller=nav, camera_index=0)

    assert nav.finish_reason == "arrived"
    assert capture.reads == 3
    assert capture.released


def test_loop_stops_after_repeated_read_failures(monkeypatch, fake_gui):
    capture = FakeCapture(frames_ok=False)
    monkeypatch.setattr(cam, "_open_camera", lambda idx, backend: capture)
    monkeypatch.setattr(cam, "MAX_CONSECUTIVE_READ_FAILURES", 5)

    cam.run_camera_loop(FakeDetector([]), camera_index=0)

    assert capture.reads == 5
    assert capture.released
