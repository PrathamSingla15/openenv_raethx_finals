"""Viz D — per-bar cumulative log-alpha vs equal-weight buy-and-hold.

The most important plot for showing that reflection actually taught the
agent to outperform the do-nothing benchmark. For each model, plot two
lines: cumulative log-alpha (agent − B&H) over the 119-bar test episode,
baseline (red dashed) vs final iter (green solid).

Positive values mean the agent is ahead of B&H at that bar.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt

from scripts.visualize.data_loader import (
    MODEL_LABELS,
    MODEL_RUNS,
    PROJECT_ROOT,
    equal_weight_bnh_series,
    load_advance_day_rows,
    load_model_iters,
)


def _agent_minus_bnh(run_dir: Path) -> tuple[list[int], list[float]]:
    rows = load_advance_day_rows(run_dir)
    if not rows:
        return [], []
    dates = [r["current_date"] for r in rows]
    bnh = equal_weight_bnh_series(dates)
    initial_value = bnh[0]
    cum_alphas = []
    for v_after, b in zip([float(r["portfolio_value"]) for r in rows], bnh):
        if v_after <= 0 or b <= 0:
            cum_alphas.append(cum_alphas[-1] if cum_alphas else 0.0)
            continue
        cum_alphas.append(math.log(v_after / initial_value) - math.log(b / initial_value))
    return list(range(len(rows))), cum_alphas


def render(out: Path) -> None:
    fig, axes = plt.subplots(1, len(MODEL_RUNS), figsize=(7.0 * len(MODEL_RUNS), 5.0), sharey=True)
    if len(MODEL_RUNS) == 1:
        axes = [axes]

    for ax, model_id in zip(axes, MODEL_RUNS.keys()):
        rows = load_model_iters(model_id)
        complete = [r for r in rows if r.status == "complete"]
        if len(complete) < 2:
            ax.set_title(f"{MODEL_LABELS[model_id]}\n(<2 completed iters)")
            continue
        baseline = complete[0]
        final = complete[-1]

        x_b, y_b = _agent_minus_bnh(baseline.run_dir)
        x_f, y_f = _agent_minus_bnh(final.run_dir)

        ax.axhline(0, color="black", linewidth=0.7, linestyle="-", alpha=0.5)
        ax.plot(x_b, y_b, "--", color="#d62728", linewidth=1.8,
                label=f"baseline (iter 0)  ROI {baseline.roi_pct:+.2f}%")
        ax.plot(x_f, y_f, "-", color="#2ca02c", linewidth=2.2,
                label=f"final (iter {final.iter})  ROI {final.roi_pct:+.2f}%")

        ax.fill_between(x_b, 0, y_b, where=[v < 0 for v in y_b],
                        alpha=0.10, color="#d62728")
        ax.fill_between(x_f, 0, y_f, where=[v > 0 for v in y_f],
                        alpha=0.15, color="#2ca02c")

        ax.set_xlabel("bar index (0 = first trading day, 119 = last)")
        ax.set_ylabel("cumulative log-alpha  (agent − equal-weight B&H)")
        ax.set_title(MODEL_LABELS[model_id], fontsize=12, weight="bold")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="lower left", framealpha=0.95, fontsize=9)

    fig.suptitle(
        "Cumulative log-alpha vs equal-weight buy-and-hold — baseline vs final iter",
        fontsize=13, weight="bold", y=1.00,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    out = PROJECT_ROOT / "docs" / "figures" / "bar_alpha_vs_bnh.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    render(out)
    print("wrote", out)
