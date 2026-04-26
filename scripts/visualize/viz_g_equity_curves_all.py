"""Viz G - all-iter equity curves vs equal-weight buy-and-hold.

Plots portfolio equity for every completed reflection iteration plus a B&H
reference on a single axes. Sequential color ramp (red baseline -> dark
green final iter) so the reader sees the climb across iterations at a
glance. Designed to be the headline chart at the top of the writeup.
"""

from __future__ import annotations

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


def _equity_series(run_dir: Path) -> tuple[list[int], list[float], list[str]]:
    rows = load_advance_day_rows(run_dir)
    if not rows:
        return [], [], []
    values = [float(r["portfolio_value"]) for r in rows]
    dates = [r["current_date"] for r in rows]
    return list(range(len(rows))), values, dates


# Sequential palette: warm red baseline -> cool dark green final iter.
_PALETTE_FULL = [
    "#c62828",  # red
    "#ef6c00",  # orange
    "#f9a825",  # amber
    "#689f38",  # light green
    "#388e3c",  # green
    "#1b5e20",  # dark green
]


def _palette_for(n: int) -> list[str]:
    """Sample n colors evenly from the warm-to-cool palette."""

    if n <= len(_PALETTE_FULL):
        idx = np.linspace(0, len(_PALETTE_FULL) - 1, n).round().astype(int)
        return [_PALETTE_FULL[int(i)] for i in idx]
    return _PALETTE_FULL


def render(out: Path, model_id: str = "qwen-qwen3-32b-groq") -> None:
    rows = [r for r in load_model_iters(model_id) if r.status == "complete"]
    if len(rows) < 2:
        raise RuntimeError(f"Need >= 2 completed iterations, got {len(rows)}")

    palette = _palette_for(len(rows))

    # B&H reference is computed from the longest-running iter's dates.
    longest = max(rows, key=lambda r: len(load_advance_day_rows(r.run_dir)))
    _, _, ref_dates = _equity_series(longest.run_dir)
    bnh = equal_weight_bnh_series(ref_dates) if ref_dates else []

    fig, ax = plt.subplots(figsize=(14.5, 6.8))

    # B&H curve.
    if bnh:
        ax.plot(
            range(len(bnh)),
            bnh,
            color="#0d47a1",
            linewidth=2.2,
            linestyle="--",
            alpha=0.9,
            zorder=10,
            label=f"equal-weight buy-and-hold  ·  {(bnh[-1]/bnh[0]-1)*100:+.2f}%",
        )

    # Iteration curves: baseline dashed, intermediate iters thinner & semi-
    # transparent, final iter thick & on top so the reader's eye lands there.
    n_rows = len(rows)
    for i, row in enumerate(rows):
        x, y, _ = _equity_series(row.run_dir)
        if not y:
            continue
        is_final = i == n_rows - 1
        is_baseline = i == 0
        color = palette[i]

        if is_final:
            linewidth = 3.0
            linestyle = "-"
            alpha = 1.0
            zorder = 14
        elif is_baseline:
            linewidth = 2.0
            linestyle = "--"
            alpha = 0.95
            zorder = 8
        else:
            linewidth = 1.6
            linestyle = "-"
            alpha = 0.85
            zorder = 5

        label = (
            f"baseline (iter 0)  ·  {row.roi_pct:+.2f}%"
            if is_baseline
            else f"iter {row.iter}{'  ★' if is_final else '       '}  ·  {row.roi_pct:+.2f}%"
        )
        ax.plot(
            x,
            y,
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=alpha,
            zorder=zorder,
            label=label,
        )

    # Soft starting-cash reference line.
    ax.axhline(100_000, color="#9ca3af", linestyle=":", linewidth=1.1, alpha=0.7, zorder=2)

    # Axis cosmetics.
    ax.set_xlabel("bar index   (0 = first trading day, 119 = last)", fontsize=11)
    ax.set_ylabel("portfolio equity", fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:,.1f}k"))
    ax.grid(True, axis="both", alpha=0.22)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#bbb")
    ax.spines["bottom"].set_color("#bbb")
    ax.tick_params(colors="#555", labelsize=10)

    # X-axis padding so the right edge has room for the legend annotations.
    longest_n = max(len(_equity_series(r.run_dir)[1]) for r in rows)
    ax.set_xlim(-2, longest_n + 4)

    # Title with two-line story.
    final = rows[-1]
    final_roi_str = f"{final.roi_pct:+.2f}%"
    base_roi_str = f"{rows[0].roi_pct:+.2f}%"
    bnh_str = f"{(bnh[-1]/bnh[0]-1)*100:+.2f}%" if bnh else "n/a"
    ax.set_title(
        f"Equity curves across reflection iterations  ·  {MODEL_LABELS[model_id]}\n"
        f"Baseline {base_roi_str}   →   iter {final.iter} {final_roi_str}   "
        f"(equal-weight B&H reference: {bnh_str})",
        fontsize=13.0,
        weight="bold",
        pad=16,
        loc="left",
    )

    # Legend outside-right, ordered baseline -> final iter, B&H reference
    # last so the warm-to-cool color ramp inside the box reads as a
    # progression. Placed outside the data area so curves are never crowded.
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        bnh_idx = next((i for i, lab in enumerate(labels) if "buy-and-hold" in lab), None)
        if bnh_idx is not None:
            order = [i for i in range(len(handles)) if i != bnh_idx] + [bnh_idx]
            handles = [handles[i] for i in order]
            labels = [labels[i] for i in order]
        leg = ax.legend(
            handles,
            labels,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            framealpha=0.0,
            fontsize=10,
            edgecolor="none",
            handlelength=2.6,
            borderpad=0.8,
            labelspacing=0.55,
        )
        for text in leg.get_texts():
            text.set_color("#222")

    # Subtle "$100k start" annotation in the lower-left corner, off-curve.
    ax.annotate(
        "$100k start",
        xy=(0, 100_000),
        xytext=(2, 100_300),
        fontsize=9,
        color="#888",
        va="bottom",
        ha="left",
        zorder=3,
    )

    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    import sys

    model_id = sys.argv[1] if len(sys.argv) > 1 else "qwen-qwen3-32b-groq"
    suffix = "" if model_id == "qwen-qwen3-32b-groq" else "_glm"
    out = PROJECT_ROOT / "docs" / "figures" / f"equity_curves_all_iters{suffix}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    render(out, model_id=model_id)
    print("wrote", out)
