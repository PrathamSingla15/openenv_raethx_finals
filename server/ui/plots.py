"""Interactive Plotly figures for the TradeBench UI.

Reuses the loaders from ``scripts/visualize/data_loader`` so every chart is
backed by the same artifacts that produced the static PNGs in ``docs/figures``.
Each builder returns a ``plotly.graph_objects.Figure`` ready for
``gr.Plot()``.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scripts.visualize.data_loader import (
    MODEL_LABELS,
    MODEL_RUNS,
    PROJECT_ROOT,
    IterRow,
    equal_weight_bnh_series,
    load_action_counts,
    load_advance_day_rows,
    load_model_iters,
    replay_reward_components,
)

# Editorial palette synced with server/ui/static/tokens.css. Cyan is the
# protagonist, amber is the "before" / baseline accent, lime is the "after"
# / converged accent. Crimson is reserved for the bar-alpha baseline trough.
COLOR_CYAN = "#22D3EE"
COLOR_CYAN_DEEP = "#0E7490"
COLOR_AMBER = "#F59E0B"
COLOR_LIME = "#84CC16"
COLOR_CRIMSON = "#E11D48"
COLOR_PAPER = "#0A0E13"
COLOR_PANEL = "#11161D"
COLOR_GRID = "#1F2731"
COLOR_AXIS = "#3A424E"
COLOR_TEXT = "#E6E8EB"
COLOR_MUTED = "#7A8693"

QWEN_ID = "qwen-qwen3-32b-groq"
GLM_ID = "zai-glm-5.1-together"

_FONT_BODY = "IBM Plex Sans, ui-sans-serif, system-ui, sans-serif"
_FONT_MONO = "JetBrains Mono, SF Mono, Menlo, monospace"


def _base_layout(*, title: str, height: int = 420) -> dict:
    """Shared dark-canvas layout. Title is set on figure level so hovermode
    can stay 'x unified' for line charts."""

    return dict(
        title=dict(
            text=f"<span style='font-family:Fraunces,serif;font-weight:500;'>{title}</span>",
            font=dict(color=COLOR_TEXT, size=15, family=_FONT_BODY),
            x=0.0,
            xanchor="left",
            pad=dict(l=8, t=4, b=4),
        ),
        paper_bgcolor=COLOR_PAPER,
        plot_bgcolor=COLOR_PANEL,
        font=dict(family=_FONT_BODY, color=COLOR_TEXT, size=12),
        margin=dict(l=64, r=24, t=56, b=56),
        height=height,
        hoverlabel=dict(
            bgcolor=COLOR_PANEL,
            bordercolor=COLOR_CYAN,
            font=dict(family=_FONT_MONO, color=COLOR_TEXT, size=12),
        ),
        xaxis=dict(
            gridcolor=COLOR_GRID,
            zerolinecolor=COLOR_GRID,
            linecolor=COLOR_AXIS,
            tickfont=dict(color=COLOR_MUTED, family=_FONT_MONO, size=11),
            title=dict(font=dict(color=COLOR_MUTED, size=12)),
        ),
        yaxis=dict(
            gridcolor=COLOR_GRID,
            zerolinecolor=COLOR_GRID,
            linecolor=COLOR_AXIS,
            tickfont=dict(color=COLOR_MUTED, family=_FONT_MONO, size=11),
            title=dict(font=dict(color=COLOR_MUTED, size=12)),
        ),
        legend=dict(
            bgcolor="rgba(17,22,29,0.75)",
            bordercolor=COLOR_GRID,
            borderwidth=1,
            font=dict(color=COLOR_TEXT, size=11),
        ),
    )


def _completed(rows: list[IterRow]) -> list[IterRow]:
    return [r for r in rows if r.status == "complete"]


def _action_counts_for(row: IterRow) -> dict[str, int]:
    counts = load_action_counts(row.run_dir)
    out = {k: v for k, v in counts.items() if not k.startswith("view_")}
    out["view_*"] = sum(v for k, v in counts.items() if k.startswith("view_"))
    return out


# ---------- Plot 1: combined trajectory ----------


def reward_trajectory() -> go.Figure:
    """Both models' score_normalized over reflection iterations."""

    fig = go.Figure()
    palette = {QWEN_ID: COLOR_CYAN, GLM_ID: COLOR_AMBER}
    dashes = {QWEN_ID: "solid", GLM_ID: "dot"}
    for model_id in (QWEN_ID, GLM_ID):
        rows = _completed(load_model_iters(model_id))
        if not rows:
            continue
        actions = [_action_counts_for(r) for r in rows]
        x = [r.iter for r in rows]
        y = [r.score_normalized for r in rows]
        roi = [r.roi_pct for r in rows]
        po = [a.get("place_order", 0) for a in actions]
        sx = [a.get("sandbox_exec", 0) for a in actions]
        vw = [a.get("view_*", 0) for a in actions]
        custom = list(zip(roi, po, sx, vw))
        color = palette[model_id]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers+text",
                name=MODEL_LABELS[model_id],
                line=dict(color=color, width=2.4, dash=dashes[model_id]),
                marker=dict(
                    size=10,
                    color=color,
                    line=dict(color=COLOR_PAPER, width=2),
                ),
                text=[f"{v:.4f}" for v in y],
                textposition="top center",
                textfont=dict(family=_FONT_MONO, color=color, size=11),
                customdata=custom,
                hovertemplate=(
                    f"<b>{MODEL_LABELS[model_id]}</b><br>"
                    "iter %{x}<br>"
                    "score_normalized: <b>%{y:.4f}</b><br>"
                    "ROI: %{customdata[0]:+.2f}%%<br>"
                    "place_order: %{customdata[1]}<br>"
                    "sandbox_exec: %{customdata[2]}<br>"
                    "view_*: %{customdata[3]}<extra></extra>"
                ),
            )
        )

    layout = _base_layout(
        title="Reward trajectory  ·  score_normalized per reflection iteration",
        height=460,
    )
    layout["xaxis"]["title"]["text"] = "reflection iteration"
    layout["xaxis"]["dtick"] = 1
    layout["yaxis"]["title"]["text"] = "score_normalized   ∈ [0, 1]"
    layout["yaxis"]["range"] = [0.605, 0.665]
    layout["hovermode"] = "x unified"
    fig.update_layout(**layout)
    return fig


# ---------- Plot 2: per-model ROI + score ----------


def roi_score_combined(model_id: str = QWEN_ID) -> go.Figure:
    """Twin-axis: ROI bars (left) + score_normalized line (right)."""

    rows = _completed(load_model_iters(model_id))
    if not rows:
        return go.Figure()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    color_bar = COLOR_LIME if model_id == QWEN_ID else COLOR_AMBER
    color_line = COLOR_CYAN

    fig.add_trace(
        go.Bar(
            x=[r.iter for r in rows],
            y=[r.roi_pct for r in rows],
            name="ROI",
            marker=dict(color=color_bar, line=dict(color=COLOR_PAPER, width=1)),
            text=[f"{r.roi_pct:+.2f}%" for r in rows],
            textposition="inside",
            textfont=dict(family=_FONT_MONO, color="#0A0E13", size=11),
            hovertemplate="iter %{x}<br>ROI: <b>%{y:+.2f}%%</b><extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=[r.iter for r in rows],
            y=[r.score_normalized for r in rows],
            name="score_normalized",
            mode="lines+markers",
            line=dict(color=color_line, width=2.4),
            marker=dict(size=9, color=color_line, line=dict(color=COLOR_PAPER, width=2)),
            hovertemplate="iter %{x}<br>score: <b>%{y:.4f}</b><extra></extra>",
        ),
        secondary_y=True,
    )

    layout = _base_layout(
        title=f"ROI and score per reflection iteration  ·  {MODEL_LABELS[model_id]}",
        height=420,
    )
    layout["xaxis"]["title"]["text"] = "reflection iteration"
    layout["xaxis"]["dtick"] = 1
    layout["bargap"] = 0.32
    layout["hovermode"] = "x unified"
    fig.update_layout(**layout)
    fig.update_yaxes(
        title_text="ROI (%)",
        secondary_y=False,
        gridcolor=COLOR_GRID,
        linecolor=COLOR_AXIS,
        tickfont=dict(color=COLOR_MUTED, family=_FONT_MONO, size=11),
        title_font=dict(color=COLOR_MUTED, size=12),
        zerolinecolor=COLOR_GRID,
    )
    fig.update_yaxes(
        title_text="score_normalized",
        secondary_y=True,
        gridcolor="rgba(0,0,0,0)",
        linecolor=COLOR_AXIS,
        tickfont=dict(color=COLOR_MUTED, family=_FONT_MONO, size=11),
        title_font=dict(color=COLOR_MUTED, size=12),
    )
    return fig


# ---------- Plot 3: per-bar alpha vs B&H ----------


def bar_alpha_vs_bnh(model_id: str = QWEN_ID) -> go.Figure:
    """Cumulative log-alpha (agent minus B&H) per bar, baseline vs final iter."""

    rows = _completed(load_model_iters(model_id))
    if len(rows) < 2:
        return go.Figure()
    base = rows[0]
    final = rows[-1]
    base_breakdown = replay_reward_components(base.run_dir)
    final_breakdown = replay_reward_components(final.run_dir)
    if not base_breakdown or not final_breakdown:
        return go.Figure()

    base_x = [b["bar"] for b in base_breakdown]
    base_y = [b["cum_alpha"] for b in base_breakdown]
    final_x = [b["bar"] for b in final_breakdown]
    final_y = [b["cum_alpha"] for b in final_breakdown]

    fig = go.Figure()

    # Shaded zero baseline + the closed-gap region (final >= baseline) as
    # a translucent area between the two curves where the green line is above.
    fig.add_trace(
        go.Scatter(
            x=base_x,
            y=base_y,
            mode="lines",
            name=f"baseline (iter 0)",
            line=dict(color=COLOR_CRIMSON, width=2.2, dash="dash"),
            hovertemplate=(
                "bar %{x}<br>"
                "<b>baseline</b><br>"
                "cum log-alpha: <b>%{y:+.4f}</b><extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=final_x,
            y=final_y,
            mode="lines",
            name=f"final (iter {final.iter})",
            line=dict(color=COLOR_LIME, width=2.4),
            fill="tonexty",
            fillcolor="rgba(132,204,22,0.10)",
            hovertemplate=(
                "bar %{x}<br>"
                f"<b>iter {final.iter}</b><br>"
                "cum log-alpha: <b>%{y:+.4f}</b><extra></extra>"
            ),
        )
    )

    # Trough callout for baseline.
    if base_y:
        trough_idx = min(range(len(base_y)), key=lambda i: base_y[i])
        fig.add_annotation(
            x=base_x[trough_idx],
            y=base_y[trough_idx],
            text=f"baseline trough · {base_y[trough_idx]:+.3f}",
            arrowhead=0,
            arrowsize=1,
            arrowwidth=1,
            arrowcolor=COLOR_CRIMSON,
            ax=40,
            ay=-50,
            font=dict(family=_FONT_MONO, color=COLOR_CRIMSON, size=11),
            bgcolor="rgba(17,22,29,0.85)",
            bordercolor=COLOR_CRIMSON,
            borderwidth=1,
            borderpad=4,
        )
    if final_y:
        last_idx = len(final_y) - 1
        fig.add_annotation(
            x=final_x[last_idx],
            y=final_y[last_idx],
            text=f"iter {final.iter} ends · {final_y[last_idx]:+.3f}",
            arrowhead=0,
            arrowcolor=COLOR_LIME,
            ax=-40,
            ay=-32,
            font=dict(family=_FONT_MONO, color=COLOR_LIME, size=11),
            bgcolor="rgba(17,22,29,0.85)",
            bordercolor=COLOR_LIME,
            borderwidth=1,
            borderpad=4,
        )

    fig.add_hline(y=0, line=dict(color=COLOR_AXIS, width=1, dash="dot"))

    layout = _base_layout(
        title=f"Per-bar cumulative log-alpha vs B&H  ·  {MODEL_LABELS[model_id]}",
        height=440,
    )
    layout["xaxis"]["title"]["text"] = "bar (test episode)"
    layout["yaxis"]["title"]["text"] = "cumulative log-alpha vs equal-weight B&H"
    layout["hovermode"] = "x unified"
    fig.update_layout(**layout)
    return fig


# ---------- Plot 4: action mix evolution ----------


def action_mix(model_id: str = QWEN_ID) -> go.Figure:
    """Grouped bars per iter; three tracked actions."""

    rows = _completed(load_model_iters(model_id))
    if not rows:
        return go.Figure()
    counts = [_action_counts_for(r) for r in rows]
    iters = [r.iter for r in rows]
    labels = [("baseline" if r.iter == 0 else f"iter {r.iter}") for r in rows]

    tracked = [
        ("place_order", COLOR_LIME),
        ("sandbox_exec", COLOR_AMBER),
        ("view_*", COLOR_CYAN),
    ]
    fig = go.Figure()
    for action, color in tracked:
        ys = [c.get(action, 0) for c in counts]
        fig.add_trace(
            go.Bar(
                x=labels,
                y=ys,
                name=action,
                marker=dict(color=color, line=dict(color=COLOR_PAPER, width=1)),
                text=[str(v) if v > 0 else "" for v in ys],
                textposition="outside",
                textfont=dict(family=_FONT_MONO, color=color, size=11),
                hovertemplate=(
                    f"<b>{action}</b><br>"
                    "%{x}<br>"
                    "count: <b>%{y}</b><extra></extra>"
                ),
            )
        )

    layout = _base_layout(
        title=f"Behavioral action mix per iteration  ·  {MODEL_LABELS[model_id]}",
        height=440,
    )
    layout["xaxis"]["title"]["text"] = "reflection iteration"
    layout["yaxis"]["title"]["text"] = "count per test episode (119 bars)"
    layout["barmode"] = "group"
    layout["bargap"] = 0.18
    layout["bargroupgap"] = 0.08
    fig.update_layout(**layout)
    return fig


# ---------- Plot 5: reward components baseline vs final ----------


def reward_components(model_id: str = QWEN_ID) -> go.Figure:
    """Mean per-bar contribution per component, baseline vs final iter, sorted by |delta|."""

    rows = _completed(load_model_iters(model_id))
    if len(rows) < 2:
        return go.Figure()
    base = rows[0]
    final = rows[-1]
    base_b = replay_reward_components(base.run_dir)
    final_b = replay_reward_components(final.run_dir)
    if not base_b or not final_b:
        return go.Figure()

    components = ["c_alpha", "c_return", "c_drawdown", "c_solvency",
                  "c_efficiency", "c_diversity", "c_consistency"]

    def mean_of(rows_, key: str) -> float:
        vals = [r[key] for r in rows_ if key in r]
        return sum(vals) / max(len(vals), 1)

    pairs = []
    for c in components:
        b = mean_of(base_b, c)
        f = mean_of(final_b, c)
        pairs.append((c, b, f, abs(f - b)))
    pairs.sort(key=lambda t: t[3], reverse=True)

    cats = [p[0] for p in pairs]
    base_vals = [p[1] for p in pairs]
    final_vals = [p[2] for p in pairs]
    deltas = [p[2] - p[1] for p in pairs]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=cats,
            y=base_vals,
            name="baseline (iter 0)",
            marker=dict(color=COLOR_AMBER, line=dict(color=COLOR_PAPER, width=1)),
            customdata=[[b] for b in base_vals],
            hovertemplate=(
                "<b>%{x}</b><br>"
                "baseline mean: <b>%{y:.3f}</b><extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Bar(
            x=cats,
            y=final_vals,
            name=f"final (iter {final.iter})",
            marker=dict(color=COLOR_LIME, line=dict(color=COLOR_PAPER, width=1)),
            customdata=list(zip(final_vals, deltas)),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "final mean: <b>%{customdata[0]:.3f}</b><br>"
                "Δ vs baseline: <b>%{customdata[1]:+.3f}</b><extra></extra>"
            ),
        )
    )

    # Delta annotations only for components with |delta| >= 0.005.
    for c, _, _, abs_d in pairs:
        delta = next(p[2] - p[1] for p in pairs if p[0] == c)
        if abs_d < 0.005:
            continue
        y = max(b for b, f in zip(base_vals, final_vals) if base_vals[cats.index(c)] == b or final_vals[cats.index(c)] == f)
        idx = cats.index(c)
        peak = max(base_vals[idx], final_vals[idx])
        fig.add_annotation(
            x=cats[idx],
            y=peak + 0.04,
            text=f"<b>{delta:+.3f}</b>",
            showarrow=False,
            font=dict(
                family=_FONT_MONO,
                color=COLOR_LIME if delta > 0 else COLOR_AMBER,
                size=11,
            ),
            bgcolor="rgba(17,22,29,0.75)",
            bordercolor=COLOR_GRID,
            borderwidth=1,
            borderpad=3,
        )

    layout = _base_layout(
        title=f"Reward components  ·  baseline vs iter {final.iter}  ·  {MODEL_LABELS[model_id]}",
        height=440,
    )
    layout["xaxis"]["title"]["text"] = "component (sorted by |Δ|)"
    layout["yaxis"]["title"]["text"] = "mean per-bar contribution  ∈ [0, 1]"
    layout["yaxis"]["range"] = [0, 1.08]
    layout["barmode"] = "group"
    layout["bargap"] = 0.22
    layout["bargroupgap"] = 0.06
    fig.update_layout(**layout)
    return fig


# ---------- Plot 6: 7-component reward stack (educational, on Reward tab) ----------


def reward_recipe() -> go.Figure:
    """Static educational chart: weights of the 7 components in the convex composite."""

    components = [
        ("c_alpha", 0.40, "vs equal-weight B&H"),
        ("c_return", 0.15, "cumulative log-return"),
        ("c_drawdown", 0.10, "1 - 2·min(dd, 0.5)"),
        ("c_solvency", 0.10, "sigmoid above 0.5·V0"),
        ("c_efficiency", 0.10, "exp(-2·max(0, turn-0.10))"),
        ("c_diversity", 0.05, "HHI-bounded clip"),
        ("c_consistency", 0.10, "alpha-stability"),
    ]
    palette = [COLOR_CYAN, COLOR_LIME, COLOR_AMBER, "#A78BFA",
               "#F472B6", "#FB923C", "#67E8F9"]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[c[0] for c in components],
            y=[c[1] for c in components],
            marker=dict(
                color=palette,
                line=dict(color=COLOR_PAPER, width=1),
            ),
            text=[f"{c[1]:.2f}" for c in components],
            textposition="outside",
            textfont=dict(family=_FONT_MONO, color=COLOR_TEXT, size=12),
            customdata=[[c[2]] for c in components],
            hovertemplate=(
                "<b>%{x}</b><br>"
                "weight: <b>%{y:.2f}</b><br>"
                "definition: %{customdata[0]}<extra></extra>"
            ),
        )
    )

    fig.add_annotation(
        x=0.5, y=1.06, xref="paper", yref="paper",
        text="weights sum to 1.00  ·  multiplied by {0,1} compliance gate",
        showarrow=False,
        font=dict(family=_FONT_MONO, color=COLOR_MUTED, size=11),
    )

    layout = _base_layout(
        title="Convex composite reward  ·  7 components, weights ∈ [0, 1]",
        height=380,
    )
    layout["yaxis"]["range"] = [0, 0.5]
    layout["yaxis"]["title"]["text"] = "weight"
    layout["xaxis"]["title"]["text"] = "component"
    layout["bargap"] = 0.32
    layout["showlegend"] = False
    fig.update_layout(**layout)
    return fig


# ---------- Plot 7: per-iter equity curves vs B&H ----------


# Sequential warm-to-cool palette mirroring viz_g_equity_curves_all.
_EQUITY_PALETTE = [
    "#c62828",  # red, baseline
    "#ef6c00",  # orange
    "#f9a825",  # amber
    "#689f38",  # light green
    "#388e3c",  # green
    "#1b5e20",  # dark green, final
]


def _equity_palette_for(n: int) -> list[str]:
    """Sample n colors evenly from the warm-to-cool palette."""
    if n <= 1:
        return [_EQUITY_PALETTE[0]]
    if n >= len(_EQUITY_PALETTE):
        return _EQUITY_PALETTE[: n] if n <= len(_EQUITY_PALETTE) else _EQUITY_PALETTE
    step = (len(_EQUITY_PALETTE) - 1) / (n - 1)
    return [_EQUITY_PALETTE[round(i * step)] for i in range(n)]


def _equity_series(run_dir):  # noqa: ANN001
    rows = load_advance_day_rows(run_dir)
    if not rows:
        return [], [], []
    values = [float(r["portfolio_value"]) for r in rows]
    dates = [r["current_date"] for r in rows]
    return list(range(len(rows))), values, dates


def equity_curves(model_id: str = QWEN_ID) -> go.Figure:
    """Per-iteration portfolio equity vs equal-weight B&H reference.

    Mirrors the static ``equity_curves_all_iters[*].png`` from
    ``scripts/visualize/viz_g_equity_curves_all.py`` but interactive: hover
    surfaces date + value + iter ROI per bar.
    """
    rows = _completed(load_model_iters(model_id))
    if len(rows) < 2:
        return go.Figure()

    palette = _equity_palette_for(len(rows))

    # B&H reference is computed from the longest-running iter's dates.
    longest = max(rows, key=lambda r: len(load_advance_day_rows(r.run_dir)))
    _, _, ref_dates = _equity_series(longest.run_dir)
    try:
        bnh = equal_weight_bnh_series(ref_dates) if ref_dates else []
    except Exception:
        bnh = []

    fig = go.Figure()

    if bnh:
        bnh_pct = [(v / bnh[0] - 1) * 100 for v in bnh]
        fig.add_trace(
            go.Scatter(
                x=list(range(len(bnh))),
                y=bnh,
                customdata=list(zip(ref_dates, bnh_pct)),
                mode="lines",
                name=f"equal-weight B&H · {bnh_pct[-1]:+.2f}%",
                line=dict(color="#0d47a1", width=2.4, dash="dash"),
                hovertemplate=(
                    "<b>equal-weight B&H</b><br>"
                    "bar %{x} · %{customdata[0]}<br>"
                    "value: <b>%{y:,.0f}</b><br>"
                    "return: <b>%{customdata[1]:+.2f}%%</b><extra></extra>"
                ),
            )
        )

    n_rows = len(rows)
    for i, row in enumerate(rows):
        x, y, dates = _equity_series(row.run_dir)
        if not y:
            continue
        is_final = i == n_rows - 1
        is_baseline = i == 0
        color = palette[i]
        width = 3.2 if is_final else (2.2 if is_baseline else 1.8)
        dash = "dash" if is_baseline else "solid"
        opacity = 1.0 if (is_final or is_baseline) else 0.85

        if is_baseline:
            label = f"baseline (iter 0) · {row.roi_pct:+.2f}%"
        elif is_final:
            label = f"iter {row.iter} ★ · {row.roi_pct:+.2f}%"
        else:
            label = f"iter {row.iter} · {row.roi_pct:+.2f}%"

        pct = [(v / y[0] - 1) * 100 for v in y]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                customdata=list(zip(dates, pct)),
                mode="lines",
                name=label,
                line=dict(color=color, width=width, dash=dash),
                opacity=opacity,
                hovertemplate=(
                    f"<b>iter {row.iter}{' ★' if is_final else ''}</b><br>"
                    "bar %{x} · %{customdata[0]}<br>"
                    "value: <b>%{y:,.0f}</b><br>"
                    "return: <b>%{customdata[1]:+.2f}%%</b><extra></extra>"
                ),
            )
        )

    layout = _base_layout(
        title=(
            f"Equity curves across all iterations  ·  {MODEL_LABELS[model_id]}"
        ),
        height=460,
    )
    layout["xaxis"]["title"]["text"] = "bar (test episode)"
    layout["yaxis"]["title"]["text"] = "portfolio value"
    layout["yaxis"]["tickformat"] = ",.0f"
    layout["hovermode"] = "x unified"
    fig.update_layout(**layout)
    return fig


__all__ = [
    "reward_trajectory",
    "roi_score_combined",
    "bar_alpha_vs_bnh",
    "action_mix",
    "reward_components",
    "reward_recipe",
    "equity_curves",
    "QWEN_ID",
    "GLM_ID",
]
