# tests/test_visualizer.py
# ─────────────────────────────────────────────────────────────
# Module 3 – Camera & YOLO Object Detection
# Unit tests for the visualizer module.
#
# Tests verify:
#   • draw_detections returns the same frame array
#   • draw_detections handles an empty detection list
#   • draw_status_bar returns the same frame array
#   • No exceptions raised for valid inputs
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import numpy as np
import pytest

from detection.visualizer import draw_detections, draw_status_bar


@pytest.fixture()
def blank_frame() -> np.ndarray:
    """A blank 480×640 BGR frame for testing."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture()
def sample_detections() -> list[dict]:
    return [
        {
            "class_name": "bottle",
            "class_id":   39,
            "confidence": 0.91,
            "bbox":       [50, 60, 200, 300],
        },
        {
            "class_name": "chair",
            "class_id":   56,
            "confidence": 0.87,
            "bbox":       [300, 100, 500, 400],
        },
    ]


class TestDrawDetections:
    def test_returns_same_frame_object(self, blank_frame, sample_detections):
        result = draw_detections(blank_frame, sample_detections)
        assert result is blank_frame

    def test_empty_detections_does_not_raise(self, blank_frame):
        result = draw_detections(blank_frame, [])
        assert result is blank_frame

    def test_frame_is_modified(self, sample_detections):
        """After drawing, the frame should differ from a blank one."""
        original = np.zeros((480, 640, 3), dtype=np.uint8)
        modified = original.copy()
        draw_detections(modified, sample_detections)
        # At least one pixel must have been changed
        assert not np.array_equal(original, modified)

    def test_single_detection(self, blank_frame):
        dets = [
            {
                "class_name": "person",
                "class_id":   0,
                "confidence": 0.94,
                "bbox":       [10, 10, 100, 200],
            }
        ]
        result = draw_detections(blank_frame, dets)
        assert result is blank_frame

    def test_bbox_at_frame_edge(self, blank_frame):
        """Bounding box at exact frame edge should not raise."""
        dets = [
            {
                "class_name": "bottle",
                "class_id":   39,
                "confidence": 0.80,
                "bbox":       [0, 0, 639, 479],
            }
        ]
        draw_detections(blank_frame, dets)  # no exception


class TestDrawStatusBar:
    def test_returns_same_frame_object(self, blank_frame):
        result = draw_status_bar(blank_frame, n_detections=3, fps=25.5)
        assert result is blank_frame

    def test_zero_detections(self, blank_frame):
        result = draw_status_bar(blank_frame, n_detections=0, fps=0.0)
        assert result is blank_frame

    def test_frame_is_modified(self):
        original = np.zeros((480, 640, 3), dtype=np.uint8)
        modified = original.copy()
        draw_status_bar(modified, n_detections=2, fps=30.0)
        assert not np.array_equal(original, modified)
