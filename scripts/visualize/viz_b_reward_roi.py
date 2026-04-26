"""Viz B — twin-axis: bar (ROI %) + line (score_normalized) per iter, faceted by model.

Tells the "trade-bench-internal score and real-money ROI move together" story
without conflating their scales.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from scripts.visualize.data_loader import (
    MODEL_LABELS,
    MODEL_RUNS,
    PROJECT_ROOT,
    load_model_iters,
)

_BAR = "#9ec5fe"
_BAR_EDGE = "#4a90e2"
_LINE = "#d62728"


def render(out: Path) -> None:
    n_models = len(MODEL_RUNS)
    fig, axes = plt.subplots(1, n_models, figsize=(6.5 * n_models, 5.5), sharey=False)
    if n_models == 1:
        axes = [axes]

    for ax, (model_id, _) in zip(axes, MODEL_RUNS.items()):
        rows = load_model_iters(model_id)
        complete = [r for r in rows if r.status == "complete"]
        if not complete:
            ax.set_title(MODEL_LABELS[model_id] + "\n(no completed iters)")
            continue
        xs = [r.iter for r in complete]
        rois = [r.roi_pct for r in complete]
        scores = [r.score_normalized for r in complete]

        bars = ax.bar(xs, rois, color=_BAR, edgecolor=_BAR_EDGE, linewidth=1.2,
                      label="ROI %", width=0.65)
        ax.set_xlabel("Reflection iteration")
        ax.set_ylabel("ROI % (vs initial $100,000)", color=_BAR_EDGE)
        ax.tick_params(axis="y", labelcolor=_BAR_EDGE)
        ax.set_xticks(xs)
        labels = ["baseline" if i == 0 else f"iter{i}" for i in xs]
        ax.set_xticklabels(labels)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.grid(True, axis="y", alpha=0.2)
        for b, v in zip(bars, rois):
            ax.text(b.get_x() + b.get_width() / 2, v + (0.3 if v >= 0 else -0.6),
                    f"{v:+.2f}%", ha="center", fontsize=9, color=_BAR_EDGE)

        ax2 = ax.twinx()
        ax2.plot(xs, scores, "-o", color=_LINE, linewidth=2.2, markersize=7,
                 label="score_normalized")
        ax2.set_ylabel("score_normalized", color=_LINE)
        ax2.tick_params(axis="y", labelcolor=_LINE)
        for x, s in zip(xs, scores):
            ax2.annotate(f"{s:.4f}", (x, s), textcoords="offset points",
                         xytext=(0, 10), ha="center", fontsize=8, color=_LINE)

        ax.set_title(MODEL_LABELS[model_id], fontsize=12, weight="bold")
        # Combined legend
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper left", framealpha=0.95, fontsize=9)

    fig.suptitle("ROI % and score_normalized per reflection iteration",
                 fontsize=13, weight="bold", y=1.00)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    out = PROJECT_ROOT / "docs" / "figures" / "reward_roi_combined.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    render(out)
    print("wrote", out)
