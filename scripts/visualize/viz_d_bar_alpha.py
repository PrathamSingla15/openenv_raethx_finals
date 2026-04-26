"""Viz D — per-bar cumulative log-alpha vs equal-weight buy-and-hold (per model).

Single-panel layout with milestone callouts and shaded "alpha gap closed" region:
- Baseline (red dashed) and final iter (green solid) on a single axis.
- Light shading between the two curves so the reader sees the gap.
- Two text callouts: baseline trough and final-iter plateau.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.visualize.data_loader import (
    MODEL_LABELS,
    PROJECT_ROOT,
    equal_weight_bnh_series,
    load_advance_day_rows,
    load_model_iters,
)

_BASELINE_COLOR = "#c62828"
_FINAL_COLOR = "#2e7d32"


def _agent_minus_bnh(run_dir: Path) -> tuple[list[int], list[float]]:
    rows = load_advance_day_rows(run_dir)
    if not rows:
        return [], []
    dates = [r["current_date"] for r in rows]
    bnh = equal_weight_bnh_series(dates)
    initial_value = bnh[0]
    cum = []
    for v_after, b in zip([float(r["portfolio_value"]) for r in rows], bnh):
        if v_after <= 0 or b <= 0:
            cum.append(cum[-1] if cum else 0.0)
            continue
        cum.append(math.log(v_after / initial_value) - math.log(b / initial_value))
    return list(range(len(rows))), cum


def render(out: Path, model_id: str = "qwen-qwen3-32b-groq") -> None:
    rows = [r for r in load_model_iters(model_id) if r.status == "complete"]
    if len(rows) < 2:
        raise RuntimeError(f"Need >= 2 completed iterations, got {len(rows)}")
    baseline = rows[0]
    final = rows[-1]

    x_b, y_b = _agent_minus_bnh(baseline.run_dir)
    x_f, y_f = _agent_minus_bnh(final.run_dir)

    fig, ax = plt.subplots(figsize=(13.0, 6.0))

    # Reference: zero line.
    ax.axhline(0, color="black", linewidth=0.8, linestyle="-", alpha=0.6,
               label="parity with equal-weight B&H")

    # Shading between the two curves: where the gap was reduced.
    n = min(len(y_b), len(y_f))
    xs_common = list(range(n))
    upper = [max(y_b[i], y_f[i]) for i in xs_common]
    lower = [min(y_b[i], y_f[i]) for i in xs_common]
    ax.fill_between(xs_common, lower, upper, color="#fbe9e7", alpha=0.6,
                    label="alpha gap closed by reflection")

    # Baseline (red dashed).
    ax.plot(x_b, y_b, "--", color=_BASELINE_COLOR, linewidth=2.0,
            label=f"baseline (iter 0) — final ROI {baseline.roi_pct:+.2f}%")
    # Final (green solid).
    ax.plot(x_f, y_f, "-", color=_FINAL_COLOR, linewidth=2.4,
            label=f"after iter {final.iter} — final ROI {final.roi_pct:+.2f}%")

    # Find baseline trough.
    if y_b:
        idx_trough = int(np.argmin(y_b))
        trough_x, trough_y = x_b[idx_trough], y_b[idx_trough]
        ax.annotate(
            f"baseline trough\n{trough_y:.3f} log-pts at bar {trough_x}",
            xy=(trough_x, trough_y), xycoords="data",
            xytext=(trough_x + 8, trough_y - 0.012),
            ha="left", va="top",
            fontsize=10, color=_BASELINE_COLOR, weight="bold",
            arrowprops=dict(arrowstyle="->", color=_BASELINE_COLOR, lw=1.4),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=_BASELINE_COLOR, linewidth=1.0, alpha=0.95),
        )

    # Final-iter callout.
    if y_f:
        end_x, end_y = x_f[-1], y_f[-1]
        ax.annotate(
            f"after iter {final.iter}\nholds at {end_y:.3f}",
            xy=(end_x, end_y), xycoords="data",
            xytext=(end_x - 28, end_y + 0.025),
            ha="left", va="bottom",
            fontsize=10, color=_FINAL_COLOR, weight="bold",
            arrowprops=dict(arrowstyle="->", color=_FINAL_COLOR, lw=1.4),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=_FINAL_COLOR, linewidth=1.0, alpha=0.95),
        )

    ax.set_xlabel("bar index   (0 = first trading day, 119 = last)", fontsize=11)
    ax.set_ylabel("cumulative log-alpha   (agent − equal-weight B&H)", fontsize=11)
    ax.set_title(
        f"Cumulative log-alpha vs equal-weight buy-and-hold  ·  {MODEL_LABELS[model_id]}\n"
        f"Reflection cut the alpha gap from {min(y_b):.3f} to {min(y_f):.3f} log-points "
        f"(better is closer to 0)",
        fontsize=12, weight="bold", pad=14,
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left", framealpha=0.95, fontsize=10)

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    import sys
    model_id = sys.argv[1] if len(sys.argv) > 1 else "qwen-qwen3-32b-groq"
    suffix = "" if model_id == "qwen-qwen3-32b-groq" else "_glm"
    out = PROJECT_ROOT / "docs" / "figures" / f"bar_alpha_vs_bnh{suffix}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    render(out, model_id=model_id)
    print("wrote", out)
