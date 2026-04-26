"""Viz B — twin-axis ROI bars + score line (per model).

Single-panel layout, anti-overlap label rules:
- ROI percent labels go INSIDE the bars (white text), so they don't
  collide with score-line annotations above the bars.
- Score-line markers labeled at iter 0 and final iter only.
- ROI delta callout in upper-left.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from scripts.visualize.data_loader import (
    MODEL_LABELS,
    PROJECT_ROOT,
    load_model_iters,
)

_MODELbar_color_COLORS = {
    "qwen-qwen3-32b-groq": ("#9ec5fe", "#1e88e5"),
    "zai-glm-5.1-together": ("#ffccbc", "#e64a19"),
}
_LINE_COLORS = {
    "qwen-qwen3-32b-groq": "#c62828",
    "zai-glm-5.1-together": "#1565c0",
}


def render(out: Path, model_id: str = "qwen-qwen3-32b-groq") -> None:
    bar_color, bar_edge = _MODELbar_color_COLORS[model_id]
    line_color = _LINE_COLORS[model_id]

    rows = [r for r in load_model_iters(model_id) if r.status == "complete"]
    if len(rows) < 2:
        raise RuntimeError(f"Need >= 2 completed iterations, got {len(rows)}")

    xs = [r.iter for r in rows]
    rois = [r.roi_pct for r in rows]
    scores = [r.score_normalized for r in rows]

    fig, ax = plt.subplots(figsize=(12.0, 6.0))

    # ROI bars on the left axis.
    bars = ax.bar(xs, rois, color=bar_color, edgecolor=bar_edge, linewidth=1.4,
                  width=0.6, label="ROI %")
    ax.set_xlabel("Reflection iteration", fontsize=11)
    ax.set_ylabel("ROI %  (final / initial - 1)", color=bar_edge, fontsize=11)
    ax.tick_params(axis="y", labelcolor=bar_edge)
    ax.set_xticks(xs)
    ax.set_xticklabels(["baseline"] + [f"iter {i}" for i in xs[1:]], fontsize=10)
    final_iter = xs[-1]
    ax.axhline(0, color="black", linewidth=0.6)
    ax.grid(True, axis="y", alpha=0.2)

    # ROI labels INSIDE bars, white text — no collision with score line above.
    for bar, roi in zip(bars, rois):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                f"{roi:+.2f}%", ha="center", va="center",
                fontsize=10.5, color="white", weight="bold")

    # Score line on twin axis.
    ax2 = ax.twinx()
    ax2.plot(xs, scores, "-o", color=line_color, linewidth=2.6, markersize=9,
             markerfacecolor="white", markeredgecolor=line_color, markeredgewidth=2.2,
             label="score_normalized")
    ax2.set_ylabel("score_normalized", color=line_color, fontsize=11)
    ax2.tick_params(axis="y", labelcolor=line_color)

    # Score labels at iter 0 and iter 5 only.
    ax2.annotate(f"{scores[0]:.4f}", xy=(xs[0], scores[0]),
                 xytext=(0, 14), textcoords="offset points",
                 ha="center", va="bottom", fontsize=10, color=line_color, weight="bold")
    ax2.annotate(f"{scores[-1]:.4f}", xy=(xs[-1], scores[-1]),
                 xytext=(0, -18), textcoords="offset points",
                 ha="center", va="top", fontsize=10, color=line_color, weight="bold")

    # Score-axis padding.
    span = max(scores) - min(scores)
    ax2.set_ylim(min(scores) - span * 0.25, max(scores) + span * 0.20)

    # ROI-axis padding so labels and callout don't fight.
    ax.set_ylim(0, max(rois) * 1.18)

    # Delta callout, upper-left.
    roi_first, roi_last = rois[0], rois[-1]
    score_first, score_last = scores[0], scores[-1]
    ax.annotate(
        f"ROI:    {roi_first:+.2f}% → {roi_last:+.2f}%   ({roi_last / roi_first:.1f}x absolute)\n"
        f"Score:  {score_first:.4f} → {score_last:.4f}   (+{score_last - score_first:.4f})",
        xy=(0.02, 0.96), xycoords="axes fraction",
        ha="left", va="top",
        fontsize=10, color="#222", family="DejaVu Sans Mono",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#fafafa",
                  edgecolor="#888", linewidth=1.0),
    )

    ax.set_title(
        f"ROI and score_normalized per iteration  ·  {MODEL_LABELS[model_id]}",
        fontsize=13, weight="bold", pad=14,
    )

    # Combined legend.
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="lower right", framealpha=0.95, fontsize=10)

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    import sys
    model_id = sys.argv[1] if len(sys.argv) > 1 else "qwen-qwen3-32b-groq"
    suffix = "" if model_id == "qwen-qwen3-32b-groq" else "_glm"
    out = PROJECT_ROOT / "docs" / "figures" / f"reward_roi_combined{suffix}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    render(out, model_id=model_id)
    print("wrote", out)
