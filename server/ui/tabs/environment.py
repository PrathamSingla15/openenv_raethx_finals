"""Environment tab: train/test split, daily loop, tool surface, schemas."""

from __future__ import annotations

import json

import gradio as gr

try:
    from ...models import TradeAction, TradeObservation, TradeState
except (ImportError, ValueError):  # pragma: no cover - dev fallback
    from models import TradeAction, TradeObservation, TradeState  # type: ignore[no-redef]


_SPLIT_HTML = """
<div class="tb-section-eyebrow">train / test split</div>
<h2 class="tb-section-title">
    <span class="tb-faded">A single-task environment with two splits.</span>
    <code>train</code> studies, <code>test</code> scores.
</h2>

<table class="tb-table">
    <thead>
        <tr>
            <th>Split</th>
            <th>Horizon</th>
            <th>Universe</th>
            <th>Role</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><code>train</code></td>
            <td class="mono">252 bars (~1 yr)</td>
            <td>10 aliased equities</td>
            <td>In-context study packet. The full window (OHLCV plus summary stats and
                correlation matrix) is delivered at episode reset. The agent reads it,
                derives a strategy, and emits <strong>one</strong> <code>record_decision</code>.
                <em>No per-bar rollout, no reward.</em></td>
        </tr>
        <tr>
            <td><code>test</code></td>
            <td class="mono">120 bars (~6 mo)</td>
            <td>Same 10 aliases</td>
            <td>Held-out walked-bar-by-bar evaluation. Calendar-adjacent to <code>train</code>
                (starts the trading day after <code>train</code> ends). The agent steps one
                bar at a time, the composite reward emits per <code>advance_day</code>, and
                the mean per-bar reward is <code>score_normalized</code>.
                <strong>This is the score that counts.</strong></td>
        </tr>
    </tbody>
</table>

<p class="tb-section-lead tb-section-lead-mute">
A separate <code>t1</code> debug tier (60 bars, 5 assets) exists for harness
validation and prompt iteration; it is not part of the scored split. Every result
in this UI is from a single <code>test</code> rollout. The deterministic grader
emits final cumulative log-wealth, max drawdown, Sharpe, Sortino, and an avoided-
ruin boolean from the ledger event log. Fully reproducible, no LLM judge.
</p>

<p class="tb-section-lead tb-section-lead-mute">
<strong>Real-derived, aliased, date-shifted data.</strong> Test bars come from real
OHLCV (mega-cap US tickers via yfinance) baked once by
<code>scripts/build_real_dataset.py</code>. Four anti-memorization layers stack on
top: <em>(1)</em> every asset is exposed only as <code>tier_a01</code> &hellip;
<code>tier_a10</code> aliases &mdash; real ticker symbols never appear in any tool
output or sandbox file; <em>(2)</em> the source window is a randomly drawn period
from <code>[2018, today]</code>, committed once at build time and never the same
across re-builds; <em>(3)</em> the alias-to-ticker mapping is permuted per build,
so even recognizing "this looks like 2022-Q3" does not reveal which alias is
AAPL; <em>(4)</em> a &sigma;=0.0005 zero-mean Gaussian return-noise overlay prevents
exact-price recall.
</p>
"""


# Daily loop SVG. Pure-black panel with white-with-alpha hairlines and cyan
# only on the active step's accent. Matches the editorial palette.
_LOOP_HTML = """
<div class="tb-section-eyebrow">the daily loop</div>
<h2 class="tb-section-title">
    observe &rarr; model &rarr; size &rarr; place orders &rarr; advance.
    <span class="tb-faded">Only</span> <code>advance_day</code>
    <span class="tb-faded">moves the clock.</span>
</h2>

<figure class="tb-figure">
    <div class="tb-figure-head">
        <div class="tb-figure-title">daily loop &middot; 5 phases</div>
        <div class="tb-figure-meta">advance_day = 1 phase, 4 free</div>
    </div>
    <div class="tb-figure-body is-dark" style="padding: 32px 24px;">
        <svg viewBox="0 0 880 180" style="width: 100%; height: auto;" role="img" aria-label="Five-step daily loop">
            <defs>
                <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto">
                    <path d="M0,0 L10,5 L0,10 z" fill="rgba(255,255,255,0.40)"/>
                </marker>
            </defs>

            <g font-family="IBM Plex Mono, monospace" font-size="11" fill="#FFFFFF">
                <rect x="20"  y="40" width="140" height="80" fill="transparent" stroke="rgba(255,255,255,0.20)"/>
                <text x="90"  y="68" text-anchor="middle" fill="#22D3EE" font-size="10" letter-spacing="2.4">OBSERVE</text>
                <text x="90"  y="92" text-anchor="middle" fill="rgba(255,255,255,0.66)" font-size="11">view_*  (x6)</text>
                <text x="90"  y="108" text-anchor="middle" fill="rgba(255,255,255,0.40)" font-size="10">free, read-only</text>

                <rect x="180" y="40" width="140" height="80" fill="transparent" stroke="rgba(255,255,255,0.20)"/>
                <text x="250" y="68" text-anchor="middle" fill="#22D3EE" font-size="10" letter-spacing="2.4">MODEL</text>
                <text x="250" y="92" text-anchor="middle" fill="rgba(255,255,255,0.66)" font-size="11">sandbox_exec</text>
                <text x="250" y="108" text-anchor="middle" fill="rgba(255,255,255,0.40)" font-size="10">no net, seccomp</text>

                <rect x="340" y="40" width="140" height="80" fill="rgba(34,211,238,0.08)" stroke="#22D3EE" stroke-width="1"/>
                <text x="410" y="68" text-anchor="middle" fill="#22D3EE" font-size="10" letter-spacing="2.4">SIZE</text>
                <text x="410" y="92" text-anchor="middle" fill="#FFFFFF" font-size="11">record_decision</text>
                <text x="410" y="108" text-anchor="middle" fill="#FFFFFF" font-size="10">required per bar</text>

                <rect x="500" y="40" width="140" height="80" fill="transparent" stroke="rgba(255,255,255,0.20)"/>
                <text x="570" y="68" text-anchor="middle" fill="#22D3EE" font-size="10" letter-spacing="2.4">ORDERS</text>
                <text x="570" y="92" text-anchor="middle" fill="rgba(255,255,255,0.66)" font-size="11">place_order</text>
                <text x="570" y="108" text-anchor="middle" fill="rgba(255,255,255,0.40)" font-size="10">queue, no same-bar</text>

                <rect x="660" y="40" width="200" height="80" fill="rgba(34,211,238,0.08)" stroke="#22D3EE" stroke-width="1"/>
                <text x="760" y="68" text-anchor="middle" fill="#22D3EE" font-size="10" letter-spacing="2.4">ADVANCE</text>
                <text x="760" y="92" text-anchor="middle" fill="#FFFFFF" font-size="11">next-open fills &middot; settle</text>
                <text x="760" y="108" text-anchor="middle" fill="#FFFFFF" font-size="10">composite reward emitted</text>
            </g>

            <g stroke="rgba(255,255,255,0.40)" stroke-width="1" fill="none" marker-end="url(#arr)">
                <line x1="160" y1="80" x2="180" y2="80"/>
                <line x1="320" y1="80" x2="340" y2="80"/>
                <line x1="480" y1="80" x2="500" y2="80"/>
                <line x1="640" y1="80" x2="660" y2="80"/>
            </g>

            <text x="440" y="160" text-anchor="middle" font-family="IBM Plex Mono, monospace"
                  font-size="10" fill="rgba(255,255,255,0.40)" letter-spacing="2">
                only ADVANCE_DAY moves t &rarr; t+1.   every other tool is free-cost, read-only or queue-only.
            </text>
        </svg>
    </div>
</figure>
"""


_TOOLS_HTML = """
<div class="tb-section-eyebrow">tool surface &middot; 11 tools</div>
<h2 class="tb-section-title">
    <span class="tb-faded">Six read-only views, three order ops, one
    time-advancer, one sandboxed Python.</span>
</h2>

<table class="tb-table">
    <thead>
        <tr><th>Tool</th><th>Purpose</th><th>Advances time?</th></tr>
    </thead>
    <tbody>
        <tr><td class="mono">view_universe</td><td>Tradable assets at <code>as_of_date</code> (date-gated)</td><td>No</td></tr>
        <tr><td class="mono">view_time</td><td>Current date, next session, bars remaining</td><td>No</td></tr>
        <tr><td class="mono">view_portfolio</td><td>Cash, positions, marks, total value</td><td>No</td></tr>
        <tr><td class="mono">view_orders</td><td>Queued + open ledger orders</td><td>No</td></tr>
        <tr><td class="mono">view_constraints</td><td>Long-only / leverage caps / sizing rules</td><td>No</td></tr>
        <tr><td class="mono">view_episode_metrics</td><td>Running score, drawdown, Sharpe (date-gated)</td><td>No</td></tr>
        <tr><td class="mono">record_decision</td><td>Log regime / edge / conviction + scanned reasoning</td><td>No</td></tr>
        <tr><td class="mono">place_order</td><td>Queue a buy/sell (no same-bar fill)</td><td>No</td></tr>
        <tr><td class="mono">cancel_order</td><td>Cancel a queued / open order by id</td><td>No</td></tr>
        <tr><td class="mono"><strong>advance_day</strong></td><td>Execute queued, settle, emit composite reward</td><td><strong>Yes</strong></td></tr>
        <tr><td class="mono">sandbox_exec</td><td>Python/shell in seccomp + no-net Docker</td><td>No</td></tr>
    </tbody>
</table>
"""


def _schema_block(title: str, model_cls) -> str:
    schema = json.dumps(model_cls.model_json_schema(), indent=2)
    return f"""
<details class="tb-schema">
    <summary>{title}</summary>
    <pre>{schema}</pre>
</details>
"""


def render() -> None:
    gr.HTML(_SPLIT_HTML)
    gr.HTML(_LOOP_HTML)
    gr.HTML(_TOOLS_HTML)
    gr.HTML(
        '<div class="tb-section-eyebrow">pydantic schemas</div>'
        '<h2 class="tb-section-title">'
        '<span class="tb-faded">Wire-level contracts.</span> '
        'Auto-rendered from <code>models.py</code>.</h2>'
        + _schema_block("TradeAction &middot; action envelope", TradeAction)
        + _schema_block("TradeObservation &middot; per-step response", TradeObservation)
        + _schema_block("TradeState &middot; episode metadata (read via /state)", TradeState),
    )
