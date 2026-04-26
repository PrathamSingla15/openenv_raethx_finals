"""Viz C — stacked-card prompt comparison with diff highlights.

Top card: initial system prompt (baseline). Lines that disappear in the
final prompt are washed in light red.
Bottom card: final system prompt after iter 5. Lines added during the
reflection loop are washed in light green.

The diff is computed at line granularity using ``difflib.SequenceMatcher``;
this is more readable than character-level for system prompts.
"""

from __future__ import annotations

import difflib
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

from scripts.visualize.data_loader import ARTIFACTS_ROOT, MODEL_RUNS, PROJECT_ROOT, load_prompt

# For Viz C we always show Qwen's evolution (the model with a complete 5-iter trajectory).
_BASELINE_RUN = MODEL_RUNS["qwen-qwen3-32b-groq"][0][1]

# The iter-5 prompt lives in the SECOND reflection root, where iter_03 inside
# corresponds to the 5th cumulative iteration (the loop resumed from iter_02).
_ITER5_PROMPT_PATH = (
    ARTIFACTS_ROOT / "reflection_20260426T032444Z" / "iter_03__reflection" / "new_system_prompt.txt"
)

_WRAP_WIDTH = 110
_FONT_SIZE = 7.0
_LINE_HEIGHT = 0.012  # in axes units per text line
_TITLE_HEIGHT = 0.04
_PAD_X = 0.015
_PAD_Y = 0.015

_COL_REMOVED_FILL = "#ffe0e0"
_COL_REMOVED_EDGE = "#cc6666"
_COL_ADDED_FILL = "#e0ffe0"
_COL_ADDED_EDGE = "#66cc66"
_COL_TITLE_FILL_TOP = "#fff0f0"
_COL_TITLE_FILL_BOT = "#f0fff0"
_COL_CARD_EDGE = "#888"


def _wrap_lines(text: str, width: int = _WRAP_WIDTH) -> list[str]:
    out: list[str] = []
    for paragraph in text.split("\n"):
        if not paragraph:
            out.append("")
            continue
        wrapped = textwrap.wrap(paragraph, width=width, break_long_words=False, break_on_hyphens=False)
        out.extend(wrapped if wrapped else [""])
    return out


def _diff_line_marks(initial_lines: list[str], final_lines: list[str]) -> tuple[set[int], set[int]]:
    """Return (removed_in_initial_idx, added_in_final_idx)."""
    matcher = difflib.SequenceMatcher(None, initial_lines, final_lines)
    removed: set[int] = set()
    added: set[int] = set()
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "delete":
            removed.update(range(i1, i2))
        elif tag == "insert":
            added.update(range(j1, j2))
        elif tag == "replace":
            removed.update(range(i1, i2))
            added.update(range(j1, j2))
    return removed, added


def _draw_card(
    fig,
    *,
    title: str,
    body_lines: list[str],
    highlight_idx: set[int],
    title_fill: str,
    highlight_fill: str,
    highlight_edge: str,
    y_top: float,
    y_bottom: float,
):
    """Draw a single card spanning [y_bottom, y_top] in figure coords."""
    ax = fig.add_axes([_PAD_X, y_bottom, 1.0 - 2 * _PAD_X, y_top - y_bottom])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Card outer rectangle
    card = FancyBboxPatch(
        (0.0, 0.0),
        1.0,
        1.0,
        boxstyle="round,pad=0.005,rounding_size=0.012",
        linewidth=1.0,
        edgecolor=_COL_CARD_EDGE,
        facecolor="white",
        transform=ax.transAxes,
    )
    ax.add_patch(card)

    # Title bar at top (height = 0.06 of card)
    title_h = 0.06
    title_box = Rectangle(
        (0.0, 1.0 - title_h),
        1.0, title_h,
        facecolor=title_fill,
        edgecolor=_COL_CARD_EDGE,
        linewidth=0.8,
        transform=ax.transAxes,
    )
    ax.add_patch(title_box)
    ax.text(
        0.012, 1.0 - title_h / 2,
        title,
        fontsize=10, weight="bold", va="center", ha="left",
        family="DejaVu Sans",
        transform=ax.transAxes,
    )

    # Body region: y from 0.005 (bottom margin) to 1.0 - title_h - 0.005
    body_top = 1.0 - title_h - 0.005
    body_bottom = 0.005
    body_height = body_top - body_bottom
    n_lines = len(body_lines)
    if n_lines == 0:
        return
    line_h_axes = body_height / max(n_lines, 1)

    # Pre-compute groups of consecutive highlight indices for cleaner shading
    if highlight_idx:
        sorted_idx = sorted(highlight_idx)
        groups: list[tuple[int, int]] = []
        run_start = sorted_idx[0]
        run_prev = sorted_idx[0]
        for i in sorted_idx[1:]:
            if i == run_prev + 1:
                run_prev = i
            else:
                groups.append((run_start, run_prev))
                run_start = i
                run_prev = i
        groups.append((run_start, run_prev))

        for g_start, g_end in groups:
            # Convert line indices to axes y-coords (text drawn top-down).
            y_top_g = body_top - g_start * line_h_axes
            y_bot_g = body_top - (g_end + 1) * line_h_axes
            ax.add_patch(Rectangle(
                (0.005, y_bot_g),
                0.99, y_top_g - y_bot_g,
                facecolor=highlight_fill,
                edgecolor=highlight_edge,
                linewidth=0.5,
                alpha=0.85,
                transform=ax.transAxes,
            ))

    # Render lines
    for i, line in enumerate(body_lines):
        y = body_top - (i + 0.5) * line_h_axes
        ax.text(
            0.012, y, line,
            fontsize=_FONT_SIZE,
            family="DejaVu Sans Mono",
            va="center", ha="left",
            transform=ax.transAxes,
        )


def render(out: Path) -> None:
    initial_text = load_prompt(ARTIFACTS_ROOT / "runs" / _BASELINE_RUN)
    final_text = _ITER5_PROMPT_PATH.read_text(encoding="utf-8")

    initial_lines = _wrap_lines(initial_text)
    final_lines = _wrap_lines(final_text)
    removed_idx, added_idx = _diff_line_marks(initial_lines, final_lines)

    # Figure size auto-scales to total line count (with floor + cap)
    total_lines = max(len(initial_lines), len(final_lines))
    height_in = max(14.0, min(28.0, 6.0 + total_lines * 0.07))
    fig = plt.figure(figsize=(13.0, height_in))

    # Header band
    fig.text(0.5, 0.985,
             "System-prompt evolution — Qwen3-32B reflection loop (baseline → iter 5)",
             ha="center", fontsize=13, weight="bold")
    fig.text(0.5, 0.965,
             "Red wash = lines removed during reflection · Green wash = lines added during reflection",
             ha="center", fontsize=10, style="italic", color="#555")

    # Two cards stacked: top = initial, bottom = final
    band_top = 0.95
    band_bottom = 0.02
    gap = 0.012
    card_h = (band_top - band_bottom - gap) / 2

    initial_y_top = band_top
    initial_y_bot = band_top - card_h
    final_y_top = initial_y_bot - gap
    final_y_bot = final_y_top - card_h

    _draw_card(
        fig,
        title=f"Initial system prompt (baseline iter 0) — {len(initial_lines)} lines",
        body_lines=initial_lines,
        highlight_idx=removed_idx,
        title_fill=_COL_TITLE_FILL_TOP,
        highlight_fill=_COL_REMOVED_FILL,
        highlight_edge=_COL_REMOVED_EDGE,
        y_top=initial_y_top,
        y_bottom=initial_y_bot,
    )
    _draw_card(
        fig,
        title=f"Final system prompt (after iter 5) — {len(final_lines)} lines",
        body_lines=final_lines,
        highlight_idx=added_idx,
        title_fill=_COL_TITLE_FILL_BOT,
        highlight_fill=_COL_ADDED_FILL,
        highlight_edge=_COL_ADDED_EDGE,
        y_top=final_y_top,
        y_bottom=final_y_bot,
    )

    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    out = PROJECT_ROOT / "docs" / "figures" / "prompt_evolution.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    render(out)
    print("wrote", out)
