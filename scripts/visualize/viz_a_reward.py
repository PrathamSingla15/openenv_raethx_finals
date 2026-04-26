"""Viz A — score_normalized over reflection iterations.

Two render modes:
- combined: both Qwen and GLM lines on one chart for the headline plot.
- single-model (Qwen-only or GLM-only): one line, anti-overlap label rules.
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

_MODEL_COLORS = {
    "qwen-qwen3-32b-groq": ("#0d47a1", "#1565c0"),
    "zai-glm-5.1-together": ("#bf360c", "#e64a19"),
}
_BASELINE = "#888888"


def _render_single(model_id: str, out: Path) -> None:
    rows = [r for r in load_model_iters(model_id) if r.status == "complete"]
    if len(rows) < 2:
        raise RuntimeError(f"Need at least 2 completed iterations, got {len(rows)}")

    xs = [r.iter for r in rows]
    ys = [r.score_normalized for r in rows]
    baseline = ys[0]
    final = ys[-1]
    delta = final - baseline
    final_iter = xs[-1]

    line_color, accent_color = _MODEL_COLORS[model_id]

    fig, ax = plt.subplots(figsize=(12.0, 6.0))

    # Reference: baseline horizontal line.
    ax.axhline(baseline, color=_BASELINE, linewidth=1.0, linestyle="--", alpha=0.7,
               label=f"baseline {baseline:.4f}")

    # Main line.
    ax.plot(xs, ys, "-o", color=line_color, linewidth=2.6, markersize=9,
            markerfacecolor="white", markeredgecolor=line_color, markeredgewidth=2.2,
            label=MODEL_LABELS[model_id])

    # Highlight markers at iter 0 and final iter with filled accent dots.
    ax.plot([xs[0], xs[-1]], [ys[0], ys[-1]], "o",
            color=accent_color, markersize=11, zorder=5)

    # Score annotations at endpoints only.
    ax.annotate(f"{baseline:.4f}", xy=(xs[0], ys[0]),
                xytext=(0, -22), textcoords="offset points",
                ha="center", va="top", fontsize=11, color=line_color, weight="bold")
    ax.annotate(f"{final:.4f}", xy=(xs[-1], ys[-1]),
                xytext=(0, 14), textcoords="offset points",
                ha="center", va="bottom", fontsize=11, color=line_color, weight="bold")

    ax.annotate(
        f"+{delta:.4f} over {final_iter} iters\n({delta / baseline * 100:.1f}% relative gain)",
        xy=(0.02, 0.96), xycoords="axes fraction",
        ha="left", va="top",
        fontsize=11, weight="bold", color=line_color,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                  edgecolor=line_color, linewidth=1.2),
    )

    ax.set_xlabel("Reflection iteration", fontsize=11)
    ax.set_ylabel("score_normalized   (mean per-bar reward, test phase)", fontsize=11)
    ax.set_xticks(xs)
    ax.set_xticklabels(["baseline\n(iter 0)"] + [f"iter {i}" for i in xs[1:]], fontsize=10)
    ax.set_title(f"Reflection-loop reward evolution  ·  {MODEL_LABELS[model_id]}",
                 fontsize=13, weight="bold", pad=14)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", framealpha=0.95, fontsize=10)

    span = max(ys) - min(ys)
    ax.set_ylim(min(ys) - span * 0.15, max(ys) + span * 0.18)

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _render_combined(out: Path) -> None:
    """Both Qwen and GLM on one chart, for the headline reward-evolution plot."""
    fig, ax = plt.subplots(figsize=(12.0, 6.0))

    all_ys: list[float] = []
    for model_id in MODEL_RUNS:
        rows = [r for r in load_model_iters(model_id) if r.status == "complete"]
        if len(rows) < 2:
            continue
        xs = [r.iter for r in rows]
        ys = [r.score_normalized for r in rows]
        all_ys.extend(ys)
        line_color, accent_color = _MODEL_COLORS[model_id]
        ax.plot(xs, ys, "-o", color=line_color, linewidth=2.4, markersize=8,
                markerfacecolor="white", markeredgecolor=line_color, markeredgewidth=2.0,
                label=f"{MODEL_LABELS[model_id]}  ({ys[0]:.4f} → {ys[-1]:.4f})")
        # Endpoint accent dots.
        ax.plot([xs[0], xs[-1]], [ys[0], ys[-1]], "o",
                color=accent_color, markersize=10, zorder=5)
        # Endpoint labels.
        ax.annotate(f"{ys[0]:.4f}", xy=(xs[0], ys[0]),
                    xytext=(-8, 4), textcoords="offset points",
                    ha="right", va="bottom", fontsize=10, color=line_color, weight="bold")
        ax.annotate(f"{ys[-1]:.4f}", xy=(xs[-1], ys[-1]),
                    xytext=(8, -4), textcoords="offset points",
                    ha="left", va="top", fontsize=10, color=line_color, weight="bold")

    ax.set_xlabel("Reflection iteration", fontsize=11)
    ax.set_ylabel("score_normalized   (mean per-bar reward, test phase)", fontsize=11)
    ax.set_xticks([0, 1, 2, 3, 4, 5])
    ax.set_xticklabels(["baseline", "iter 1", "iter 2", "iter 3", "iter 4", "iter 5"], fontsize=10)
    ax.set_title("Reflection-loop reward evolution  ·  Qwen3-32B vs GLM-5.1",
                 fontsize=13, weight="bold", pad=14)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", framealpha=0.95, fontsize=10, title="Model  (baseline → final)")

    if all_ys:
        span = max(all_ys) - min(all_ys)
        ax.set_ylim(min(all_ys) - span * 0.15, max(all_ys) + span * 0.18)

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render(out: Path, model_id: str | None = None) -> None:
    """Dispatch: combined if model_id is None, single-model otherwise."""
    if model_id is None:
        _render_combined(out)
    else:
        _render_single(model_id, out)


if __name__ == "__main__":
    out = PROJECT_ROOT / "docs" / "figures" / "reward_evolution.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    render(out)
    print("wrote", out)
