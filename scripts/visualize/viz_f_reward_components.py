"""Viz F — mean reward-component contribution: baseline vs final iter.

Replays each run's trajectory through ``compute_composite_reward`` (the live,
post-redesign convex composite) to obtain per-bar values for the seven named
components plus the compliance gate. Bars-mean values are plotted as a
grouped bar chart, baseline vs final iter, faceted by model.

Surfaces *which* of the seven reward axes the agent actually moved.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.visualize.data_loader import (
    MODEL_LABELS,
    MODEL_RUNS,
    PROJECT_ROOT,
    load_model_iters,
    replay_reward_components,
)

_COMPONENTS = ["c_alpha", "c_return", "c_drawdown", "c_solvency",
               "c_efficiency", "c_diversity", "c_consistency"]


def _component_means(run_dir: Path) -> dict[str, float]:
    breakdowns = replay_reward_components(run_dir)
    if not breakdowns:
        return {k: 0.0 for k in _COMPONENTS}
    return {k: sum(b[k] for b in breakdowns) / len(breakdowns) for k in _COMPONENTS}


def render(out: Path) -> None:
    n_models = len(MODEL_RUNS)
    fig, axes = plt.subplots(1, n_models, figsize=(7.0 * n_models, 5.5), sharey=True)
    if n_models == 1:
        axes = [axes]

    bar_w = 0.35
    x = np.arange(len(_COMPONENTS))

    for ax, model_id in zip(axes, MODEL_RUNS.keys()):
        rows = load_model_iters(model_id)
        complete = [r for r in rows if r.status == "complete"]
        if len(complete) < 2:
            ax.set_title(f"{MODEL_LABELS[model_id]}\n(<2 completed iters)")
            continue
        baseline = complete[0]
        final = complete[-1]

        m_b = _component_means(baseline.run_dir)
        m_f = _component_means(final.run_dir)

        ax.bar(x - bar_w / 2, [m_b[c] for c in _COMPONENTS], bar_w,
               label="baseline (iter 0)", color="#d62728", alpha=0.75)
        ax.bar(x + bar_w / 2, [m_f[c] for c in _COMPONENTS], bar_w,
               label=f"final (iter {final.iter})", color="#2ca02c", alpha=0.85)

        # Annotate deltas
        for xi, c in zip(x, _COMPONENTS):
            delta = m_f[c] - m_b[c]
            color = "#2ca02c" if delta >= 0 else "#d62728"
            top = max(m_b[c], m_f[c]) + 0.02
            ax.text(xi, top, f"{delta:+.3f}", ha="center",
                    fontsize=8, color=color, weight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(_COMPONENTS, rotation=20, ha="right", fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("mean per-bar contribution (component value, [0,1])")
        ax.set_title(MODEL_LABELS[model_id], fontsize=12, weight="bold")
        ax.grid(True, axis="y", alpha=0.25)
        ax.axhline(0.5, linestyle="--", color="black", linewidth=0.5, alpha=0.4)
        ax.legend(loc="lower right", framealpha=0.95, fontsize=9)

    fig.suptitle("Mean per-bar reward components: baseline vs final iter",
                 fontsize=13, weight="bold", y=1.00)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    out = PROJECT_ROOT / "docs" / "figures" / "reward_components_baseline_vs_final.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    render(out)
    print("wrote", out)
