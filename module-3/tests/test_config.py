# tests/test_config.py
# ─────────────────────────────────────────────────────────────
# Module 3 – Camera & YOLO Object Detection
# Unit tests for the config module.
#
# Verifies that config values have the expected types and
# that environment variable overrides work correctly.
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import os
from importlib import reload
from unittest.mock import patch


class TestConfigDefaults:
    """Default config values must have correct types."""

    def test_confidence_threshold_is_float(self):
        import config.config as cfg
        assert isinstance(cfg.CONFIDENCE_THRESHOLD, float)

    def test_confidence_threshold_range(self):
        import config.config as cfg
        assert 0.0 <= cfg.CONFIDENCE_THRESHOLD <= 1.0

    def test_inference_img_size_is_int(self):
        import config.config as cfg
        assert isinstance(cfg.INFERENCE_IMG_SIZE, int)
        assert cfg.INFERENCE_IMG_SIZE > 0

    def test_camera_index_is_int(self):
        import config.config as cfg
        assert isinstance(cfg.CAMERA_INDEX, int)
        assert cfg.CAMERA_INDEX >= 0

    def test_yolo_model_path_is_str(self):
        import config.config as cfg
        assert isinstance(cfg.YOLO_MODEL_PATH, str)
        assert len(cfg.YOLO_MODEL_PATH) > 0

    def test_inference_device_is_str(self):
        import config.config as cfg
        assert isinstance(cfg.INFERENCE_DEVICE, str)

    def test_quit_key_is_int(self):
        import config.config as cfg
        assert isinstance(cfg.QUIT_KEY, int)

    def test_log_every_n_frames_is_non_negative(self):
        import config.config as cfg
        assert cfg.LOG_EVERY_N_FRAMES >= 0


class TestConfigEnvOverrides:
    """Environment variables should override config defaults."""

    def test_confidence_threshold_env_override(self):
        with patch.dict(os.environ, {"CONFIDENCE_THRESHOLD": "0.75"}):
            import config.config as cfg
            reload(cfg)
            assert cfg.CONFIDENCE_THRESHOLD == pytest.approx(0.75)

    def test_camera_index_env_override(self):
        with patch.dict(os.environ, {"CAMERA_INDEX": "2"}):
            import config.config as cfg
            reload(cfg)
            assert cfg.CAMERA_INDEX == 2

    def test_yolo_model_path_env_override(self):
        with patch.dict(os.environ, {"YOLO_MODEL_PATH": "yolov8s.pt"}):
            import config.config as cfg
            reload(cfg)
            assert cfg.YOLO_MODEL_PATH == "yolov8s.pt"


import pytest  # noqa: E402 — import after class definitions to keep tests clean
