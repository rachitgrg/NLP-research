# tests/test_obstacle_detector.py
# ─────────────────────────────────────────────────────────────
# Module 3 – Unit tests for ObstacleDetector
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import pytest

from guidance.obstacle_detector import ObstacleDetector


@pytest.fixture()
def detector() -> ObstacleDetector:
    return ObstacleDetector(
        enabled=True,
        corridor_left=0.25,
        corridor_right=0.75,
        corridor_y_min=0.35,
        min_area_ratio=0.05,
        obstacle_classes=("chair", "couch", "dining table", "person"),
    )


class TestObstacleDetectorHazards:
    """Verify detection of hazardous obstacles blocking the walking corridor."""

    def test_chair_in_center_path_triggers_hazard(self, detector):
        # 1000x1000 frame: chair from x=350 to x=650, y=400 to y=850
        # Area = 300*450 = 135,000 / 1,000,000 = 0.135 (> 0.05)
        # Center x = 500 (in corridor 250..750), y2 = 850 (> 350)
        detections = [
            {"class_name": "chair", "confidence": 0.88, "bbox": [350, 400, 650, 850]}
        ]
        report = detector.evaluate(detections, 1000, 1000, target_class="bottle")

        assert report.has_hazard is True
        assert report.primary_hazard is not None
        assert report.primary_hazard.class_name == "chair"
        assert "chair" in report.primary_hazard.warning_text.lower()

    def test_target_class_itself_is_not_treated_as_obstacle(self, detector):
        # User is searching for a chair! That chair should not be flagged as a hazard.
        detections = [
            {"class_name": "chair", "confidence": 0.88, "bbox": [350, 400, 650, 850]}
        ]
        report = detector.evaluate(detections, 1000, 1000, target_class="chair")

        assert report.has_hazard is False
        assert len(report.hazards) == 0

    def test_non_obstacle_classes_ignored(self, detector):
        # Cell phone or cup in corridor should not trigger collision warnings
        detections = [
            {"class_name": "cell phone", "confidence": 0.90, "bbox": [400, 400, 600, 600]},
            {"class_name": "cup", "confidence": 0.85, "bbox": [450, 450, 550, 550]},
        ]
        report = detector.evaluate(detections, 1000, 1000, target_class="bottle")
        assert report.has_hazard is False

    def test_obstacle_outside_walking_corridor_ignored(self, detector):
        # Chair far off on the left (x: 10 to 180, outside 250..750 corridor)
        detections = [
            {"class_name": "chair", "confidence": 0.91, "bbox": [10, 400, 180, 800]}
        ]
        report = detector.evaluate(detections, 1000, 1000, target_class="bottle")
        assert report.has_hazard is False

    def test_obstacle_too_small_ignored(self, detector):
        # Chair far in background (tiny box 20x20 in 1000x1000 = 0.0004 area ratio)
        detections = [
            {"class_name": "chair", "confidence": 0.70, "bbox": [490, 490, 510, 510]}
        ]
        report = detector.evaluate(detections, 1000, 1000, target_class="bottle")
        assert report.has_hazard is False

    def test_obstacle_high_in_air_ignored(self, detector):
        # Object near top of frame (y2 = 250 < 350 floor corridor)
        detections = [
            {"class_name": "person", "confidence": 0.85, "bbox": [350, 50, 650, 250]}
        ]
        report = detector.evaluate(detections, 1000, 1000, target_class="bottle")
        assert report.has_hazard is False


class TestObstacleDetectorDetourDirections:
    """Verify intelligent detour directions around obstacles."""

    def test_obstacle_on_left_advises_step_right(self, detector):
        # Centroid cx = 350 (< 500) -> on the left
        detections = [
            {"class_name": "chair", "confidence": 0.89, "bbox": [260, 400, 440, 850]}
        ]
        report = detector.evaluate(detections, 1000, 1000, target_class="bottle")
        assert report.has_hazard is True
        assert report.primary_hazard.detour_direction == "right"
        assert "step slightly right" in report.primary_hazard.warning_text.lower()

    def test_obstacle_on_right_advises_step_left(self, detector):
        # Centroid cx = 650 (> 500) -> on the right
        detections = [
            {"class_name": "person", "confidence": 0.92, "bbox": [560, 400, 740, 850]}
        ]
        report = detector.evaluate(detections, 1000, 1000, target_class="bottle")
        assert report.has_hazard is True
        assert report.primary_hazard.detour_direction == "left"
        assert "step slightly left" in report.primary_hazard.warning_text.lower()

    def test_wide_obstacle_advises_stop(self, detector):
        # Wide table spanning x: 100 to 900 (span = 0.80 > 0.65)
        detections = [
            {"class_name": "dining table", "confidence": 0.88, "bbox": [100, 400, 900, 800]}
        ]
        report = detector.evaluate(detections, 1000, 1000, target_class="bottle")
        assert report.has_hazard is True
        assert report.primary_hazard.detour_direction == "stop"
        assert "stop" in report.primary_hazard.warning_text.lower()


class TestObstacleDetectorEdgeCases:
    """Verify edge cases and disabled modes."""

    def test_disabled_detector_returns_empty(self):
        disabled = ObstacleDetector(enabled=False)
        detections = [
            {"class_name": "chair", "confidence": 0.99, "bbox": [300, 300, 700, 700]}
        ]
        report = disabled.evaluate(detections, 1000, 1000)
        assert report.has_hazard is False

    def test_empty_detections_returns_empty(self, detector):
        report = detector.evaluate([], 1000, 1000)
        assert report.has_hazard is False

    def test_invalid_frame_dimensions_returns_empty(self, detector):
        detections = [{"class_name": "chair", "confidence": 0.99, "bbox": [10, 10, 50, 50]}]
        report = detector.evaluate(detections, 0, 1000)
        assert report.has_hazard is False


# ── Objects holding / supporting the target are not blockers ────

def _person(bbox):
    return {"class_name": "person", "class_id": 0, "confidence": 0.85, "bbox": bbox}


def test_person_holding_target_is_not_a_hazard():
    det = ObstacleDetector(enabled=True)
    person = _person([160, 112, 591, 480])
    report = det.evaluate([person], 640, 480, target_class="bottle", target_bbox=[301, 111, 385, 328])
    assert report.has_hazard is False


def test_person_not_containing_target_is_still_a_hazard():
    det = ObstacleDetector(enabled=True)
    person = _person([160, 112, 591, 480])
    report = det.evaluate([person], 640, 480, target_class="bottle", target_bbox=[10, 10, 60, 100])
    assert report.has_hazard is True


def test_without_target_bbox_behaviour_unchanged():
    det = ObstacleDetector(enabled=True)
    report = det.evaluate([_person([160, 112, 591, 480])], 640, 480, target_class="bottle")
    assert report.has_hazard is True


def test_furniture_containing_target_is_still_a_hazard():
    """'Bottle on chair' vs 'chair in front of bottle' is ambiguous in 2D: keep warning."""
    det = ObstacleDetector(enabled=True)
    chair = {"class_name": "chair", "class_id": 1, "confidence": 0.9, "bbox": [280, 200, 520, 480]}
    report = det.evaluate([chair], 640, 480, target_class="bottle", target_bbox=[380, 250, 420, 330])
    assert report.has_hazard is True
