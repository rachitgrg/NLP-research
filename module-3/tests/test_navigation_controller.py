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
        assert "not found" in step.spoken_instruction.lower()

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
