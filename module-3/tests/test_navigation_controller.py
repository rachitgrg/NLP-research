# tests/test_navigation_controller.py
# ─────────────────────────────────────────────────────────────
# Module 3 – Unit tests for NavigationController
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import pytest

from audio_feedback.tts_speaker import TTSSpeaker
from guidance.obstacle_detector import ObstacleDetector
from guidance.spatial_locator import HorizontalZone, SpatialLocator
from matching.target_matcher import TargetMatcher
from navigation.navigation_controller import NavigationController
from navigation.navigation_state import NavigationState


@pytest.fixture()
def controller() -> NavigationController:
    speaker = TTSSpeaker(enabled=True, use_mock=True)
    ctrl = NavigationController(
        target_name="bottle",
        matcher=TargetMatcher(),
        locator=SpatialLocator(arrival_area_th=0.22),
        obstacle_detector=ObstacleDetector(),
        speaker=speaker,
        searching_cooldown_s=0.1,
    )
    yield ctrl
    speaker.stop()


class TestNavigationControllerStates:
    """Verify state transitions of the assistive navigation engine."""

    def test_initial_state_is_searching(self, controller):
        assert controller.state == NavigationState.SEARCHING

    def test_target_absent_stays_searching(self, controller):
        detections = [
            {"class_name": "person", "confidence": 0.90, "bbox": [10, 10, 50, 50]}
        ]
        step = controller.process_frame(detections, 1000, 1000)
        assert step.state == NavigationState.SEARCHING
        assert step.target_found is False
        assert "not seen" in step.spoken_instruction.lower()

    def test_target_detected_transitions_to_guiding(self, controller):
        # Bottle at center, small area
        detections = [
            {"class_name": "bottle", "confidence": 0.88, "bbox": [450, 450, 550, 550]}
        ]
        step = controller.process_frame(detections, 1000, 1000)
        assert step.state == NavigationState.GUIDING
        assert step.target_found is True
        assert step.spatial_location is not None
        assert step.spatial_location.zone == HorizontalZone.CENTER
        assert "straight ahead" in step.spoken_instruction.lower()

    def test_target_disappearing_transitions_to_lost(self, controller):
        # 1. Target found
        detections_found = [
            {"class_name": "bottle", "confidence": 0.88, "bbox": [450, 450, 550, 550]}
        ]
        controller.process_frame(detections_found, 1000, 1000)
        assert controller.state == NavigationState.GUIDING

        # 2. Target lost in next frame
        detections_empty = []
        step_lost = controller.process_frame(detections_empty, 1000, 1000)
        assert step_lost.state == NavigationState.LOST
        assert "lost" in step_lost.spoken_instruction.lower()

    def test_large_target_transitions_to_arrived(self, controller):
        # Large bottle (> 22% frame area)
        detections_arrived = [
            {"class_name": "bottle", "confidence": 0.95, "bbox": [200, 200, 800, 800]}
        ]
        step = controller.process_frame(detections_arrived, 1000, 1000)
        assert step.state == NavigationState.ARRIVED
        assert step.is_arrived is True
        assert "in front of you" in step.spoken_instruction.lower()
        assert "reach out" in step.spoken_instruction.lower()

    def test_hazard_blocks_path_transitions_to_obstructed(self, controller):
        # Bottle is visible straight ahead, BUT a chair is in the central walking path
        detections = [
            {"class_name": "bottle", "confidence": 0.85, "bbox": [450, 450, 550, 550]},
            {"class_name": "chair",  "confidence": 0.90, "bbox": [300, 400, 700, 850]},
        ]
        step = controller.process_frame(detections, 1000, 1000)
        assert step.state == NavigationState.OBSTRUCTED
        assert step.obstacle_report.has_hazard is True
        assert "chair" in step.spoken_instruction.lower()

    def test_unsupported_target_transitions_to_unsupported(self):
        speaker = TTSSpeaker(enabled=True, use_mock=True)
        ctrl = NavigationController(target_name="wristwatch", speaker=speaker)
        detections = [{"class_name": "clock", "confidence": 0.9, "bbox": [100, 100, 200, 200]}]
        step = ctrl.process_frame(detections, 1000, 1000)
        assert step.state == NavigationState.UNSUPPORTED
        assert "not detectable" in step.spoken_instruction.lower()
        speaker.stop()

    def test_reset_clears_state(self, controller):
        detections = [
            {"class_name": "bottle", "confidence": 0.88, "bbox": [450, 450, 550, 550]}
        ]
        controller.process_frame(detections, 1000, 1000)
        assert controller.ever_found is True

        controller.reset("cup")
        assert controller.state == NavigationState.SEARCHING
        assert controller.target_name == "cup"
        assert controller.ever_found is False


def test_persistent_hazard_does_not_flood_speech_queue():
    """60 frames (~3 s) with the same hazard must queue at most one warning."""
    speaker = TTSSpeaker(enabled=True, cooldown_s=2.0, use_mock=True)
    speaker.stop()  # keep items in the queue so we can count them
    nc = NavigationController("bottle", speaker=speaker)
    dets = [{"class_name": "chair", "class_id": 1, "confidence": 0.9, "bbox": [200, 200, 440, 470]}]
    for _ in range(60):
        step = nc.process_frame(dets, 640, 480)
    assert step.state == NavigationState.OBSTRUCTED
    assert speaker._queue.qsize() <= 1


# ── Session completion (arrival confirmation / timeout) ─────────

_BIG_BOTTLE = [{"class_name": "bottle", "class_id": 5, "confidence": 0.9, "bbox": [100, 50, 540, 450]}]
_PERSON_IN_PATH = [{"class_name": "person", "class_id": 0, "confidence": 0.9, "bbox": [180, 150, 460, 480]}]


def _controller(**kwargs) -> NavigationController:
    return NavigationController("bottle", speaker=TTSSpeaker(enabled=True, use_mock=True), **kwargs)


def test_arrival_requires_consecutive_frames():
    nc = _controller(arrival_confirm_frames=3)
    nc.process_frame(_BIG_BOTTLE, 640, 480)
    nc.process_frame(_BIG_BOTTLE, 640, 480)
    assert nc.finish_reason is None
    nc.process_frame(_BIG_BOTTLE, 640, 480)
    assert nc.finish_reason == "arrived"


def test_arrival_streak_resets_when_target_missing():
    nc = _controller(arrival_confirm_frames=2)
    nc.process_frame(_BIG_BOTTLE, 640, 480)
    nc.process_frame([], 640, 480)
    nc.process_frame(_BIG_BOTTLE, 640, 480)
    assert nc.finish_reason is None


def test_search_times_out(monkeypatch):
    import navigation.navigation_controller as mod
    clock = [1000.0]
    monkeypatch.setattr(mod.time, "monotonic", lambda: clock[0])
    nc = _controller(search_timeout_s=10)
    nc.process_frame([], 640, 480)
    clock[0] += 9
    nc.process_frame([], 640, 480)
    assert nc.finish_reason is None
    clock[0] += 2
    nc.process_frame([], 640, 480)
    assert nc.finish_reason == "timeout"


def test_timeout_not_blocked_by_persistent_obstacle(monkeypatch):
    import navigation.navigation_controller as mod
    clock = [1000.0]
    monkeypatch.setattr(mod.time, "monotonic", lambda: clock[0])
    nc = _controller(search_timeout_s=10)
    nc.process_frame(_PERSON_IN_PATH, 640, 480)
    clock[0] += 11
    nc.process_frame(_PERSON_IN_PATH, 640, 480)
    assert nc.finish_reason == "timeout"


def test_reset_clears_finish_reason():
    nc = _controller(arrival_confirm_frames=1)
    nc.process_frame(_BIG_BOTTLE, 640, 480)
    assert nc.finish_reason == "arrived"
    nc.reset()
    assert nc.finish_reason is None


def test_unsupported_target_finishes_session():
    nc = NavigationController("watch", speaker=TTSSpeaker(enabled=True, use_mock=True))
    nc.process_frame([], 640, 480)
    assert nc.finish_reason == "unsupported"


def test_bottle_held_by_person_gets_guidance_not_obstacle_warning(monkeypatch):
    """Regression: 'Stop. Person ahead.' replaced all guidance while the user held the bottle."""
    import navigation.navigation_controller as mod
    clock = [100.0]
    monkeypatch.setattr(mod.time, "monotonic", lambda: clock[0])
    nc = NavigationController("bottle", speaker=TTSSpeaker(enabled=True, use_mock=True),
                              obstacle_detector=ObstacleDetector(enabled=True), search_timeout_s=0)
    person = {"class_name": "person", "class_id": 0, "confidence": 0.85, "bbox": [160, 112, 591, 480]}
    bottle = {"class_name": "bottle", "class_id": 5, "confidence": 0.5, "bbox": [301, 111, 385, 328]}

    step = nc.process_frame([person, bottle], 640, 480)
    assert step.state == NavigationState.GUIDING
    assert "bottle" in step.spoken_instruction.lower()

    # Bottle flickers out for 1 s: the person holding it is still not an obstacle.
    clock[0] += 1.0
    step = nc.process_frame([person], 640, 480)
    assert step.state != NavigationState.OBSTRUCTED

    # Long after the bottle was last seen, the person counts as an obstacle again.
    clock[0] += 5.0
    step = nc.process_frame([person], 640, 480)
    assert step.state == NavigationState.OBSTRUCTED
