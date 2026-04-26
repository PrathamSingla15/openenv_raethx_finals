"""Viz E — behavioral action mix per reflection iteration (per model).

Single-panel layout with explicit y-headroom and inline summary box.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.visualize.data_loader import (
    MODEL_LABELS,
    PROJECT_ROOT,
    load_action_counts,
    load_model_iters,
)

_TRACKED = [
    ("place_order", "#2e7d32"),
    ("sandbox_exec", "#ef6c00"),
    ("view_*", "#6a1b9a"),
]


def _consolidate_views(counts: dict) -> dict:
    out = {k: v for k, v in counts.items() if not k.startswith("view_")}
    view_total = sum(v for k, v in counts.items() if k.startswith("view_"))
    if view_total > 0:
        out["view_*"] = view_total
    return out


def render(out: Path, model_id: str = "qwen-qwen3-32b-groq") -> None:
    rows = [r for r in load_model_iters(model_id) if r.status == "complete"]
    if not rows:
        raise RuntimeError("no completed iterations")

    per_iter = []
    labels = []
    for r in rows:
        per_iter.append(_consolidate_views(load_action_counts(r.run_dir)))
        labels.append("baseline" if r.iter == 0 else f"iter {r.iter}")

    fig, ax = plt.subplots(figsize=(13.0, 5.5))

    bar_w = 0.27
    x = np.arange(len(per_iter))
    all_heights: list[int] = []

    for i, (action, color) in enumerate(_TRACKED):
        offset = (i - 1) * bar_w
        heights = [d.get(action, 0) for d in per_iter]
        all_heights.extend(heights)
        bars = ax.bar(x + offset, heights, bar_w, color=color, label=action,
                      edgecolor="white", linewidth=0.6)
        for b, h in zip(bars, heights):
            if h <= 0:
                continue
            ax.text(b.get_x() + b.get_width() / 2, h + 0.6,
                    f"{h}", ha="center", va="bottom",
                    fontsize=9, color=color, weight="bold")

    y_max = max(all_heights) if all_heights else 1
    ax.set_ylim(0, y_max * 1.30)  # extra headroom so labels never collide with the summary box

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("count per test episode", fontsize=11)
    ax.set_title(
        f"Behavioral action mix per reflection iteration  ·  {MODEL_LABELS[model_id]}",
        fontsize=13, weight="bold", pad=18,
    )
    ax.text(0.5, 1.02,
            "advance_day and record_decision excluded — the per-bar gate forces them to ~constant",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=9.5, style="italic", color="#555")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper left", framealpha=0.95, fontsize=10)

    # Growth callout in upper-right. Use left-aligned monospace inside the
    # box (right-aligning a multi-line block looks messy when line widths vary).
    po_first = per_iter[0].get("place_order", 0)
    po_last = per_iter[-1].get("place_order", 0)
    sx_first = per_iter[0].get("sandbox_exec", 0)
    sx_last = per_iter[-1].get("sandbox_exec", 0)
    vw_first = per_iter[0].get("view_*", 0)
    vw_last = per_iter[-1].get("view_*", 0)
    summary = (
        f"baseline → iter {rows[-1].iter}\n"
        f"  place_order    {po_first:>3} → {po_last:<3}  ({po_last / max(po_first, 1):.1f}x)\n"
        f"  sandbox_exec   {sx_first:>3} → {sx_last:<3}  (front-loaded analysis)\n"
        f"  view_*         {vw_first:>3} → {vw_last:<3}  ({vw_last / max(vw_first, 1):.1f}x)"
    )
    ax.text(0.985, 0.965, summary,
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9.5, color="#222", family="DejaVu Sans Mono",
            multialignment="left",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#fafafa",
                      edgecolor="#999", linewidth=1.0))

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    import sys
    model_id = sys.argv[1] if len(sys.argv) > 1 else "qwen-qwen3-32b-groq"
    suffix = "" if model_id == "qwen-qwen3-32b-groq" else "_glm"
    out = PROJECT_ROOT / "docs" / "figures" / f"action_mix_evolution{suffix}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    render(out, model_id=model_id)
    print("wrote", out)
