import os
import csv
from pathlib import Path

def plot_results(csv_path: str, output_path: str):
    """
    Generate a 4-panel comparison bar chart from the evaluation results.csv.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for graph generation.\n"
            "Install it with:  pip install matplotlib"
        ) from exc

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Results CSV not found: {csv_path}")

    results = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)

    if not results:
        raise ValueError("Results CSV is empty.")

    models = ["sarvam", "whisper_tiny", "whisper_base"]
    model_labels = ["Sarvam AI", "Whisper Tiny", "Whisper Base"]
    colors = ["#E05C3A", "#4A90D9", "#56B4E9"]
    
    # Calculate means
    means = {m: {"wer": 0.0, "cer": 0.0, "em": 0.0, "latency": 0.0} for m in models}
    n = len(results)
    
    for r in results:
        for m in models:
            means[m]["wer"] += float(r[f"{m}_wer"])
            means[m]["cer"] += float(r[f"{m}_cer"])
            em_val = r[f"{m}_exact_match"] == "True"
            means[m]["em"] += 1.0 if em_val else 0.0
            means[m]["latency"] += float(r[f"{m}_latency_seconds"])
            
    for m in models:
        means[m]["wer"] /= n
        means[m]["cer"] /= n
        means[m]["em"] = (means[m]["em"] / n) * 100.0  # Percentage
        means[m]["latency"] /= n
        
    BG_COLOR = "#F7F9FC"
    GRID_COLOR = "#D0D7E3"

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle(
        f"Speech Recognition Benchmark (Stage 1)\n{n} samples",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    x = np.arange(len(models))
    bar_width = 0.5
    
    def style_axis(ax, title, ylabel, data_key, lower_is_better=True, is_percent=False):
        ax.set_facecolor(BG_COLOR)
        ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        
        values = [means[m][data_key] for m in models]
        bars = ax.bar(x, values, bar_width, color=colors, zorder=3)
        
        ax.set_title(title, fontsize=13, pad=10)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels, fontsize=11)
        
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_color(GRID_COLOR)
            
        for bar in bars:
            height = bar.get_height()
            fmt = f"{height:.1f}%" if is_percent else f"{height:.3f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + (max(values)*0.02 if max(values) > 0 else 0.1),
                fmt,
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                color="#333333",
            )
            
    # Panel 1: WER (Lower is better)
    style_axis(axes[0,0], "Word Error Rate (WER)", "Error Rate", "wer", lower_is_better=True)
    
    # Panel 2: CER (Lower is better)
    style_axis(axes[0,1], "Character Error Rate (CER)", "Error Rate", "cer", lower_is_better=True)
    
    # Panel 3: Exact Match (Higher is better)
    style_axis(axes[1,0], "Exact Match Rate", "Percentage (%)", "em", lower_is_better=False, is_percent=True)
    axes[1,0].set_ylim(0, 110)
    
    # Panel 4: Latency (Lower is better)
    style_axis(axes[1,1], "Average Latency", "Seconds", "latency", lower_is_better=True)
    
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=100, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"4-panel graph saved to: {out}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "results.csv")
    out_path = os.path.join(base_dir, "comparison_graph_v2.png")
    plot_results(csv_path, out_path)
