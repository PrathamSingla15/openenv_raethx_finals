"""Viz F — mean reward-component contribution: baseline vs final iter (per model).

Single-panel layout, sorted by |delta|, hide annotations for tiny deltas.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.visualize.data_loader import (
    MODEL_LABELS,
    PROJECT_ROOT,
    load_model_iters,
    replay_reward_components,
)

_BASELINE = "#c62828"
_FINAL = "#2e7d32"
_DELTA_HIDE_BELOW = 0.005

_COMPONENTS = ["c_alpha", "c_return", "c_drawdown", "c_solvency",
               "c_efficiency", "c_diversity", "c_consistency"]


def _component_means(run_dir: Path) -> dict[str, float]:
    breakdowns = replay_reward_components(run_dir)
    if not breakdowns:
        return {k: 0.0 for k in _COMPONENTS}
    return {k: sum(b[k] for b in breakdowns) / len(breakdowns) for k in _COMPONENTS}


def render(out: Path, model_id: str = "qwen-qwen3-32b-groq") -> None:
    rows = [r for r in load_model_iters(model_id) if r.status == "complete"]
    if len(rows) < 2:
        raise RuntimeError(f"Need >= 2 completed iterations, got {len(rows)}")
    baseline = rows[0]
    final = rows[-1]
    m_b = _component_means(baseline.run_dir)
    m_f = _component_means(final.run_dir)

    # Sort by absolute delta descending so changed-most components are leftmost.
    deltas = {c: m_f[c] - m_b[c] for c in _COMPONENTS}
    sorted_components = sorted(_COMPONENTS, key=lambda c: -abs(deltas[c]))

    fig, ax = plt.subplots(figsize=(12.0, 6.0))

    bar_w = 0.36
    x = np.arange(len(sorted_components))
    base_vals = [m_b[c] for c in sorted_components]
    final_vals = [m_f[c] for c in sorted_components]

    bars_b = ax.bar(x - bar_w / 2, base_vals, bar_w,
                    color=_BASELINE, alpha=0.78, edgecolor="white", linewidth=0.5,
                    label="baseline (iter 0)")
    bars_f = ax.bar(x + bar_w / 2, final_vals, bar_w,
                    color=_FINAL, alpha=0.85, edgecolor="white", linewidth=0.5,
                    label=f"after iter {final.iter}")

    # Delta annotations only when |delta| meaningful.
    for xi, c in zip(x, sorted_components):
        d = deltas[c]
        if abs(d) < _DELTA_HIDE_BELOW:
            continue
        color = _FINAL if d >= 0 else _BASELINE
        top = max(m_b[c], m_f[c]) + 0.025
        ax.text(xi, top, f"{d:+.3f}", ha="center", va="bottom",
                fontsize=11, color=color, weight="bold",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor=color, linewidth=1.0, alpha=0.95))

    ax.axhline(0.5, color="#666", linewidth=0.7, linestyle="--", alpha=0.7,
               label="neutral (0.5)")

    ax.set_xticks(x)
    ax.set_xticklabels(sorted_components, rotation=18, ha="right", fontsize=10.5)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("mean per-bar contribution   (each c_i in [0, 1])", fontsize=11)
    ax.set_title(
        f"Reward-component decomposition: baseline vs iter {final.iter}  ·  "
        f"{MODEL_LABELS[model_id]}\n"
        "Sorted left → right by |Δ| ;  components with |Δ| < 0.005 left unannotated",
        fontsize=12, weight="bold", pad=12,
    )
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="lower right", framealpha=0.95, fontsize=10)

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    import sys
    model_id = sys.argv[1] if len(sys.argv) > 1 else "qwen-qwen3-32b-groq"
    suffix = "" if model_id == "qwen-qwen3-32b-groq" else "_glm"
    out = PROJECT_ROOT / "docs" / "figures" / f"reward_components_baseline_vs_final{suffix}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    render(out, model_id=model_id)
    print("wrote", out)
