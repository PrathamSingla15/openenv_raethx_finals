"""Viz E — behavioral action mix per reflection iteration.

Grouped bars per iter for the actions that ACTUALLY change across iters:
``place_order`` (the headline "did the agent take real positions" metric),
``sandbox_exec`` ("did the agent actually compute"), and ``view_*``
("did the agent observe state"). ``advance_day`` and ``record_decision``
are excluded because the per-bar gate forces them to constants (~119 and
~135 respectively, regardless of policy quality).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.visualize.data_loader import (
    MODEL_LABELS,
    MODEL_RUNS,
    PROJECT_ROOT,
    load_action_counts,
    load_model_iters,
)

_TRACKED = [
    ("place_order", "#2ca02c"),
    ("sandbox_exec", "#ff7f0e"),
    ("view_*", "#9467bd"),
]


def _consolidate_views(counts: dict) -> dict:
    out = dict(counts)
    view_total = sum(v for k, v in out.items() if k.startswith("view_"))
    out = {k: v for k, v in out.items() if not k.startswith("view_")}
    if view_total > 0:
        out["view_*"] = view_total
    return out


def render(out: Path) -> None:
    n_models = len(MODEL_RUNS)
    fig, axes = plt.subplots(1, n_models, figsize=(7.5 * n_models, 5.0), sharey=False)
    if n_models == 1:
        axes = [axes]

    bar_w = 0.27

    for ax, model_id in zip(axes, MODEL_RUNS.keys()):
        rows = load_model_iters(model_id)
        complete = [r for r in rows if r.status == "complete"]
        if not complete:
            ax.set_title(f"{MODEL_LABELS[model_id]} — no completed iters")
            continue

        per_iter = []
        labels = []
        for r in complete:
            cnts = _consolidate_views(load_action_counts(r.run_dir))
            per_iter.append(cnts)
            labels.append("baseline" if r.iter == 0 else f"iter {r.iter}")

        x = np.arange(len(per_iter))
        for i, (action, color) in enumerate(_TRACKED):
            offset = (i - 1) * bar_w
            heights = [d.get(action, 0) for d in per_iter]
            bars = ax.bar(x + offset, heights, bar_w, color=color, label=action,
                          edgecolor="white", linewidth=0.5)
            for b, h in zip(bars, heights):
                if h > 0:
                    ax.text(b.get_x() + b.get_width() / 2, h + max(heights) * 0.025 + 0.5,
                            f"{h}", ha="center", fontsize=8, color=color, weight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("count per test episode")
        ax.set_title(MODEL_LABELS[model_id], fontsize=12, weight="bold")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(loc="upper left", framealpha=0.95, fontsize=9)

        # Add a footer caption showing the place_order delta
        if len(per_iter) > 1:
            po_first = per_iter[0].get("place_order", 0)
            po_last = per_iter[-1].get("place_order", 0)
            ax.text(0.5, -0.16,
                    f"place_order: {po_first} → {po_last}  "
                    f"({(po_last/po_first if po_first else float('inf')):.1f}× growth across reflection)",
                    transform=ax.transAxes, ha="center", fontsize=9, style="italic", color="#444")

    fig.suptitle("Behavioral action mix per reflection iteration "
                 "(advance_day & record_decision excluded — gated to ~constant)",
                 fontsize=12, weight="bold", y=1.00)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    out = PROJECT_ROOT / "docs" / "figures" / "action_mix_evolution.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    render(out)
    print("wrote", out)
