from __future__ import annotations

# evaluation/graph.py
# ─────────────────────────────────────────────────────────────
# Module 1 – Hearing & Understanding
# Generates a dual-panel comparison bar chart (PNG) comparing
# Whisper Tiny and Sarvam AI on accuracy and transcription time.
#
# Requires matplotlib:
#     pip install matplotlib
# ─────────────────────────────────────────────────────────────

from pathlib import Path


def generate_comparison_graph(results: list[dict], output_path: str) -> str:
    """
    Generate a dual-panel comparison bar chart and save it as a PNG.

    Panel 1 → Accuracy (%):   Whisper Tiny vs Sarvam AI
    Panel 2 → Avg Time (s):   Whisper Tiny vs Sarvam AI

    Args:
        results:     List of per-file result dicts, each containing at
                     minimum the keys produced by evaluate_models.py:
                     ``whisper_accuracy``, ``sarvam_accuracy``,
                     ``whisper_time_seconds``, ``sarvam_time_seconds``.
        output_path: Absolute or relative path where the PNG will be saved.
                     Parent directories are created automatically.

    Returns:
        The resolved absolute path to the saved PNG file.

    Raises:
        ImportError: If matplotlib is not installed.
        ValueError:  If ``results`` is empty.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for graph generation.\n"
            "Install it with:  pip install matplotlib"
        ) from exc

    if not results:
        raise ValueError("results list is empty — cannot generate graph.")

    n = len(results)

    whisper_acc   = [r["whisper_accuracy"]    for r in results]
    sarvam_acc    = [r["sarvam_accuracy"]     for r in results]
    whisper_times = [r["whisper_time_seconds"] for r in results]
    sarvam_times  = [r["sarvam_time_seconds"]  for r in results]

    avg_whisper_acc  = sum(whisper_acc)   / n
    avg_sarvam_acc   = sum(sarvam_acc)    / n
    avg_whisper_time = sum(whisper_times) / n
    avg_sarvam_time  = sum(sarvam_times)  / n
    avg_whisper_wer  = sum(r["whisper_wer"] for r in results) / n
    avg_sarvam_wer   = sum(r["sarvam_wer"]  for r in results) / n
    avg_whisper_cer  = sum(r["whisper_cer"] for r in results) / n
    avg_sarvam_cer   = sum(r["sarvam_cer"]  for r in results) / n

    WHISPER_COLOR = "#4A90D9"
    SARVAM_COLOR  = "#E05C3A"
    BG_COLOR      = "#F7F9FC"
    GRID_COLOR    = "#D0D7E3"

    fig, axes = plt.subplots(1, 2, figsize=(13, 8.5))
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle(
        "Speech Recognition Model Comparison\nWhisper Tiny  vs  Sarvam AI Saaras v3",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )

    models    = ["Whisper Tiny", "Sarvam AI\nSaaras v3"]
    bar_width = 0.45

    import numpy as np
    x = np.arange(2)

    def _annotate(ax, bars, fmt):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 1.2,
                fmt.format(height),
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                color="white",
            )

    # ── Panel 1: Accuracy ────────────────────────────────────────
    ax1 = axes[0]
    ax1.set_facecolor(BG_COLOR)
    ax1.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax1.set_axisbelow(True)

    acc_values = [avg_whisper_acc, avg_sarvam_acc]
    acc_colors = [WHISPER_COLOR, SARVAM_COLOR]
    bars1 = ax1.bar(x, acc_values, bar_width, color=acc_colors, zorder=3)

    subtitle_acc = (
        f"Whisper: WER={avg_whisper_wer:.3f}, CER={avg_whisper_cer:.3f}"
        f"\nSarvam:  WER={avg_sarvam_wer:.3f},  CER={avg_sarvam_cer:.3f}"
    )
    ax1.set_title(
        f"Average Accuracy\n({n} sample" + ("s" if n != 1 else "") + ")",
        fontsize=13,
        pad=10,
    )
    ax1.set_xlabel(subtitle_acc, fontsize=8.5, color="#555555", labelpad=8)
    ax1.set_ylabel("Accuracy (%)", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=10)
    ax1.set_ylim(0, 115)
    ax1.tick_params(axis="y", labelsize=10)

    for spine in ax1.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color(GRID_COLOR)

    _annotate(ax1, bars1, "{:.1f}%")

    winner = "Whisper Tiny" if avg_whisper_acc > avg_sarvam_acc else "Sarvam AI"
    ax1.text(
        0.5, 108,
        f"[WINNER]  {winner} wins on accuracy",
        ha="center",
        fontsize=9,
        color="#555555",
        style="italic",
    )

    # ── Panel 2: Transcription Time ──────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor(BG_COLOR)
    ax2.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax2.set_axisbelow(True)

    time_values = [avg_whisper_time, avg_sarvam_time]
    bars2 = ax2.bar(x, time_values, bar_width, color=acc_colors, zorder=3)

    ax2.set_title(
        f"Average Transcription Time\n({n} sample" + ("s" if n != 1 else "") + ")",
        fontsize=13,
        pad=10,
    )
    ax2.set_xlabel("(lower is faster)", fontsize=8.5, color="#555555", labelpad=8)
    ax2.set_ylabel("Time (seconds)", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=10)
    ax2.tick_params(axis="y", labelsize=10)

    for spine in ax2.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color(GRID_COLOR)

    _annotate(ax2, bars2, "{:.2f}s")

    faster = "Whisper Tiny" if avg_whisper_time < avg_sarvam_time else "Sarvam AI"
    ax2.text(
        0.5,
        max(time_values) * 1.22,
        f"[FASTER]  {faster} is faster",
        ha="center",
        fontsize=9,
        color="#555555",
        style="italic",
    )

    # ── Legend & layout ──────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(color=WHISPER_COLOR, label="Whisper Tiny  (local, CPU)"),
        mpatches.Patch(color=SARVAM_COLOR,  label="Sarvam AI Saaras v3  (cloud)"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.12),
    )

    fig.tight_layout()

    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=100, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)

    return str(out)
