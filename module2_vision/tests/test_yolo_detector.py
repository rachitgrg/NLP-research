# tests/test_yolo_detector.py
# ─────────────────────────────────────────────────────────────
# Module 3 – Camera & YOLO Object Detection
# Unit tests for YOLODetector (mocked — no real model needed).
#
# These tests verify:
#   • Detection output structure (keys, types)
#   • Confidence threshold filtering
#   • Empty detection handling
#   • class_names property
#   • BBox coordinate types (must be int)
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ── Helper: build a fake Ultralytics result ───────────────────

def _make_fake_box(
    x1: float, y1: float, x2: float, y2: float,
    class_id: int, conf: float
) -> MagicMock:
    """Return a mock object that looks like a YOLO box tensor entry."""
    box = MagicMock()
    box.xyxy = [MagicMock()]
    box.xyxy[0].tolist.return_value = [x1, y1, x2, y2]
    box.cls = [MagicMock()]
    box.cls[0].item.return_value = class_id
    box.conf = [MagicMock()]
    box.conf[0].item.return_value = conf
    return box


def _make_fake_result(boxes_data: list[tuple]) -> MagicMock:
    """
    Build a mock Ultralytics Results object.

    boxes_data : list of (x1, y1, x2, y2, class_id, conf)
    """
    result = MagicMock()
    result.names = {
        39: "bottle",
        56: "chair",
        0:  "person",
    }
    if boxes_data:
        result.boxes = [
            _make_fake_box(*bd) for bd in boxes_data
        ]
    else:
        result.boxes = []
    return result


# ── Fixtures ─────────────────────────────────────────────────

@pytest.fixture()
def detector():
    """YOLODetector with all heavy dependencies mocked out."""
    with patch("detection.yolo_detector.YOLO") as MockYOLO:
        mock_model = MagicMock()
        # Warm-up call — return empty result
        mock_model.predict.return_value = [_make_fake_result([])]
        mock_model.names = {39: "bottle", 56: "chair", 0: "person"}
        MockYOLO.return_value = mock_model

        from detection.yolo_detector import YOLODetector

        det = YOLODetector(
            model_path="yolov8n.pt",
            confidence_threshold=0.40,
            img_size=640,
            device="cpu",
        )
        # Expose the underlying mock for tests to customise
        det._mock_model = mock_model
        yield det


# ── Tests ─────────────────────────────────────────────────────

class TestYOLODetectorOutputStructure:
    """Detection dicts must have the correct keys and types."""

    def test_single_detection_keys(self, detector):
        fake_result = _make_fake_result([(10, 20, 200, 300, 39, 0.91)])
        detector._mock_model.predict.return_value = [fake_result]

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect(frame)

        assert len(detections) == 1
        det = detections[0]
        assert set(det.keys()) == {"class_name", "class_id", "confidence", "bbox"}

    def test_class_name_is_string(self, detector):
        fake_result = _make_fake_result([(10, 20, 200, 300, 39, 0.91)])
        detector._mock_model.predict.return_value = [fake_result]

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect(frame)

        assert isinstance(detections[0]["class_name"], str)
        assert detections[0]["class_name"] == "bottle"

    def test_class_id_is_int(self, detector):
        fake_result = _make_fake_result([(10, 20, 200, 300, 39, 0.91)])
        detector._mock_model.predict.return_value = [fake_result]

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect(frame)

        assert isinstance(detections[0]["class_id"], int)
        assert detections[0]["class_id"] == 39

    def test_confidence_is_float(self, detector):
        fake_result = _make_fake_result([(10, 20, 200, 300, 39, 0.91)])
        detector._mock_model.predict.return_value = [fake_result]

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect(frame)

        assert isinstance(detections[0]["confidence"], float)

    def test_bbox_is_list_of_four_ints(self, detector):
        fake_result = _make_fake_result([(10, 20, 200, 300, 39, 0.91)])
        detector._mock_model.predict.return_value = [fake_result]

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect(frame)

        bbox = detections[0]["bbox"]
        assert isinstance(bbox, list)
        assert len(bbox) == 4
        assert all(isinstance(v, int) for v in bbox)
        assert bbox == [10, 20, 200, 300]


class TestYOLODetectorMultipleObjects:
    """Multiple objects in a single frame."""

    def test_multiple_detections_count(self, detector):
        fake_result = _make_fake_result([
            (10, 20, 200, 300, 39,  0.91),
            (50, 60, 400, 500, 56,  0.87),
            (5,  10, 300, 400, 0,   0.94),
        ])
        detector._mock_model.predict.return_value = [fake_result]

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect(frame)

        assert len(detections) == 3

    def test_class_names_match(self, detector):
        fake_result = _make_fake_result([
            (10, 20, 200, 300, 39, 0.91),
            (50, 60, 400, 500, 56, 0.87),
        ])
        detector._mock_model.predict.return_value = [fake_result]

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect(frame)

        names = [d["class_name"] for d in detections]
        assert "bottle" in names
        assert "chair" in names


class TestYOLODetectorEmptyCases:
    """No detections or empty results list."""

    def test_empty_boxes_returns_empty_list(self, detector):
        fake_result = _make_fake_result([])
        detector._mock_model.predict.return_value = [fake_result]

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect(frame)

        assert detections == []

    def test_empty_results_list_returns_empty_list(self, detector):
        detector._mock_model.predict.return_value = []

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect(frame)

        assert detections == []

    def test_none_boxes_returns_empty_list(self, detector):
        fake_result = _make_fake_result([])
        fake_result.boxes = None
        detector._mock_model.predict.return_value = [fake_result]

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect(frame)

        assert detections == []


class TestYOLODetectorProperties:
    """Detector properties and accessors."""

    def test_confidence_threshold_property(self, detector):
        assert detector.confidence_threshold == 0.40

    def test_class_names_property_returns_dict(self, detector):
        names = detector.class_names
        assert isinstance(names, dict)
        assert names[39] == "bottle"


class TestYOLODetectorMissingLibrary:
    """Graceful ImportError when ultralytics is not installed."""

    def test_import_error_raised_with_clear_message(self):
        # Patch the module-level YOLO name to None to simulate missing library
        with patch("detection.yolo_detector.YOLO", None):
            with pytest.raises(ImportError, match="ultralytics"):
                from detection.yolo_detector import YOLODetector
                YOLODetector(
                    model_path="yolov8n.pt",
                    confidence_threshold=0.40,
                    img_size=640,
                    device="cpu",
                )


class TestTargetConfidenceThreshold:
    """B2: the requested class may pass at a lower confidence than others."""

    def _boxes(self, detector):
        detector._mock_model.predict.return_value = [_make_fake_result([
            (10, 10, 50, 90, 39, 0.30),   # bottle, between 0.25 and 0.40
            (60, 10, 99, 90, 56, 0.30),   # chair, same confidence
            (0, 0, 200, 300, 0, 0.80),    # person, confident
        ])]

    def test_without_target_everything_needs_base_threshold(self, detector):
        self._boxes(detector)
        names = [d["class_name"] for d in detector.detect(np.zeros((100, 100, 3), np.uint8))]
        assert names == ["person"]

    def test_target_kept_from_lower_threshold_others_unchanged(self, detector):
        detector.set_target("Bottle", 0.25)
        self._boxes(detector)
        names = [d["class_name"] for d in detector.detect(np.zeros((100, 100, 3), np.uint8))]
        assert sorted(names) == ["bottle", "person"]

    def test_predict_runs_at_lowest_threshold(self, detector):
        detector.set_target("bottle", 0.25)
        detector.detect(np.zeros((100, 100, 3), np.uint8))
        assert detector._mock_model.predict.call_args.kwargs["conf"] == 0.25

    def test_clearing_target_restores_base(self, detector):
        detector.set_target("bottle", 0.25)
        detector.set_target(None)
        self._boxes(detector)
        names = [d["class_name"] for d in detector.detect(np.zeros((100, 100, 3), np.uint8))]
        assert names == ["person"]
