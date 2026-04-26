"""Viz A — score_normalized over reflection iterations.

One line per model. In-flight rows render as a dashed segment to the last
known point so the user can see at a glance which model is still training.
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

_COLOR = {
    "qwen-qwen3-32b-groq": "#1f77b4",
    "zai-glm-5.1-together": "#d62728",
}


def render(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    for model_id in MODEL_RUNS:
        rows = load_model_iters(model_id)
        complete = [r for r in rows if r.status == "complete"]
        if not complete:
            continue
        xs = [r.iter for r in complete]
        ys = [r.score_normalized for r in complete]
        color = _COLOR[model_id]
        label = MODEL_LABELS[model_id]
        ax.plot(xs, ys, "-o", color=color, linewidth=2.2, markersize=7, label=label)
        # Annotate baseline and best
        ax.annotate(
            f"{ys[0]:.4f}",
            xy=(xs[0], ys[0]),
            xytext=(0, -16),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color=color,
        )
        if len(ys) > 1:
            ax.annotate(
                f"{ys[-1]:.4f}",
                xy=(xs[-1], ys[-1]),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                color=color,
                weight="bold",
            )
        # Trailing dashed segment if next iter is in_flight
        last_iter = xs[-1]
        next_pending = [r for r in rows if r.iter > last_iter and r.status == "in_flight"]
        if next_pending:
            next_iter = next_pending[0].iter
            ax.plot(
                [last_iter, next_iter], [ys[-1], ys[-1]],
                "--", color=color, alpha=0.4, linewidth=1.5,
            )
            ax.annotate(
                f"iter{next_iter} in flight",
                xy=(next_iter, ys[-1]),
                xytext=(2, -12),
                textcoords="offset points",
                fontsize=8,
                color=color,
                style="italic",
            )

    ax.set_xlabel("Reflection iteration")
    ax.set_ylabel("score_normalized (test phase, mean per-bar reward)")
    ax.set_xticks(range(0, 6))
    ax.set_xticklabels(["baseline\n(iter 0)", "iter 1", "iter 2", "iter 3", "iter 4", "iter 5"])
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", framealpha=0.95)
    ax.set_title("Reflection-loop reward evolution", fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    out = PROJECT_ROOT / "docs" / "figures" / "reward_evolution.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    render(out)
    print("wrote", out)
