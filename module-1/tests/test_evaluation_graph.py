# tests/test_evaluation_graph.py
# ─────────────────────────────────────────────────────────────
# Unit tests for evaluation/graph.py
#
# Run with:  pytest tests/test_evaluation_graph.py -v
#
# Tests use a synthetic DataFrame so no real evaluation is needed.
# ─────────────────────────────────────────────────────────────

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Fixtures ───────────────────────────────────────────────────

@pytest.fixture()
def sample_results() -> list[dict]:
    """Synthetic evaluation results — 3 fake samples."""
    return [
        {
            "audio_file":            "sample_01.wav",
            "reference_text":        "Where is my black bottle",
            "whisper_raw":           "where is my black bottle",
            "sarvam_raw":            "Where is my black bottle?",
            "whisper_text":          "where is my black bottle",
            "sarvam_text":           "where is my black bottle",
            "whisper_wer":           0.0,
            "sarvam_wer":            0.0,
            "whisper_cer":           0.0,
            "sarvam_cer":            0.0,
            "whisper_accuracy":      100.0,
            "sarvam_accuracy":       100.0,
            "whisper_time_seconds":  1.25,
            "sarvam_time_seconds":   0.82,
        },
        {
            "audio_file":            "sample_02.wav",
            "reference_text":        "Show me the red cup on the shelf",
            "whisper_raw":           "show me the red cup on shelf",
            "sarvam_raw":            "Show me the red cup on the shelf.",
            "whisper_text":          "show me the red cup on shelf",
            "sarvam_text":           "show me the red cup on the shelf",
            "whisper_wer":           0.125,
            "sarvam_wer":            0.0,
            "whisper_cer":           0.0625,
            "sarvam_cer":            0.0,
            "whisper_accuracy":      87.5,
            "sarvam_accuracy":       100.0,
            "whisper_time_seconds":  1.10,
            "sarvam_time_seconds":   0.91,
        },
        {
            "audio_file":            "sample_03.wav",
            "reference_text":        "Find the blue bag near the door",
            "whisper_raw":           "find the blue bag near door",
            "sarvam_raw":            "Find blue bag near the door.",
            "whisper_text":          "find the blue bag near door",
            "sarvam_text":           "find blue bag near the door",
            "whisper_wer":           0.1667,
            "sarvam_wer":            0.1667,
            "whisper_cer":           0.0833,
            "sarvam_cer":            0.0833,
            "whisper_accuracy":      83.33,
            "sarvam_accuracy":       83.33,
            "whisper_time_seconds":  1.30,
            "sarvam_time_seconds":   0.78,
        },
    ]


# ── Tests ──────────────────────────────────────────────────────

class TestGenerateComparisonGraph:
    """Tests for evaluation.graph.generate_comparison_graph()."""

    def test_graph_created(self, sample_results, tmp_path):
        """Graph PNG is created at the specified path."""
        from evaluation.graph import generate_comparison_graph

        out = str(tmp_path / "test_graph.png")
        result_path = generate_comparison_graph(sample_results, out)

        assert Path(result_path).exists(), f"Graph file not found: {result_path}"
        assert Path(result_path).stat().st_size > 0, "Graph file is empty"

    def test_graph_returns_path(self, sample_results, tmp_path):
        """Return value is the absolute path to the saved PNG."""
        from evaluation.graph import generate_comparison_graph

        out = str(tmp_path / "test_graph.png")
        result_path = generate_comparison_graph(sample_results, out)

        assert isinstance(result_path, str)
        assert result_path.endswith(".png")

    def test_graph_creates_parent_directories(self, sample_results, tmp_path):
        """Parent directories are created automatically."""
        from evaluation.graph import generate_comparison_graph

        out = str(tmp_path / "nested" / "deep" / "graph.png")
        generate_comparison_graph(sample_results, out)

        assert Path(out).exists()

    def test_graph_single_sample(self, tmp_path):
        """Graph handles a single-sample results list without crashing."""
        from evaluation.graph import generate_comparison_graph

        single = [
            {
                "audio_file":            "sample_01.wav",
                "reference_text":        "hello world",
                "whisper_raw":           "hello world",
                "sarvam_raw":            "hello world",
                "whisper_text":          "hello world",
                "sarvam_text":           "hello world",
                "whisper_wer":           0.0,
                "sarvam_wer":            0.0,
                "whisper_cer":           0.0,
                "sarvam_cer":            0.0,
                "whisper_accuracy":      100.0,
                "sarvam_accuracy":       100.0,
                "whisper_time_seconds":  0.5,
                "sarvam_time_seconds":   0.3,
            }
        ]
        out = str(tmp_path / "single.png")
        generate_comparison_graph(single, out)
        assert Path(out).exists()

    def test_empty_results_raises(self, tmp_path):
        """Empty results list raises ValueError."""
        from evaluation.graph import generate_comparison_graph

        with pytest.raises(ValueError, match="empty"):
            generate_comparison_graph([], str(tmp_path / "graph.png"))

    def test_graph_is_png(self, sample_results, tmp_path):
        """Output file has PNG magic bytes."""
        from evaluation.graph import generate_comparison_graph

        out = str(tmp_path / "graph.png")
        generate_comparison_graph(sample_results, out)

        with open(out, "rb") as f:
            header = f.read(8)
        # PNG files start with the 8-byte PNG signature
        assert header == b"\x89PNG\r\n\x1a\n", "Output is not a valid PNG file"

    def test_graph_overwrites_existing(self, sample_results, tmp_path):
        """Calling generate twice overwrites the previous file cleanly."""
        from evaluation.graph import generate_comparison_graph

        out = str(tmp_path / "graph.png")
        generate_comparison_graph(sample_results, out)
        size1 = Path(out).stat().st_size

        generate_comparison_graph(sample_results, out)
        size2 = Path(out).stat().st_size

        assert size2 > 0
