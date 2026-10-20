# tests/test_m3_pipeline_bridge.py
# ─────────────────────────────────────────────────────────────
# Module 3 – Module 1 integration bridge
# Regression tests for loading Module 1's HearingPipeline.
#
# Both modules contain a top-level package named "pipeline";
# the bridge must load Module 1's hearing_pipeline.py without
# colliding with Module 3's own pipeline package.
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import importlib.util
from unittest.mock import patch

import pytest

import pipeline.m3_pipeline as bridge

_MODULE1_DEPS = all(
    importlib.util.find_spec(name) is not None
    for name in ("sarvamai", "spacy")
)


@pytest.mark.skipif(not _MODULE1_DEPS, reason="Module 1 dependencies not installed")
def test_module1_importable_despite_pipeline_name_clash():
    assert bridge.MODULE1_AVAILABLE is True
    assert bridge._HearingPipeline.__name__ == "HearingPipeline"


def test_module3_pipeline_package_not_shadowed():
    """Module 3's own pipeline package must still resolve to module-3."""
    import pipeline
    assert "module-3" in pipeline.__file__.replace("\\", "/")


def test_run_voice_pipeline_maps_module1_result():
    class FakePipeline:
        def run(self, audio_path):
            return {
                "text": "where is my black bottle",
                "language": "hi-IN",
                "intent": "find_object",
                "object": "bottle",
                "attributes": {"color": "black"},
                "keywords": ["black", "bottle"],
            }

    with patch.object(bridge, "MODULE1_AVAILABLE", True), \
         patch.object(bridge, "_HearingPipeline", FakePipeline):
        result = bridge.run_voice_pipeline("dummy.wav")

    assert result["target"] == "bottle"
    assert result["language"] == "hi-IN"
    assert result["attributes"] == {"color": "black"}


def test_run_voice_pipeline_raises_when_module1_unavailable():
    with patch.object(bridge, "MODULE1_AVAILABLE", False):
        with pytest.raises(RuntimeError):
            bridge.run_voice_pipeline("dummy.wav")
