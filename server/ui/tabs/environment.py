"""Environment tab — task tiers, daily loop, tool surface, Pydantic schemas."""

from __future__ import annotations

import json

import gradio as gr

try:
    from ...models import TradeAction, TradeObservation, TradeState
except (ImportError, ValueError):  # pragma: no cover - dev fallback
    from models import TradeAction, TradeObservation, TradeState  # type: ignore[no-redef]


_TIERS_HTML = """
<div class="tb-section-eyebrow">three task tiers</div>
<h2 class="tb-section-title">Each tier is a deterministic episode generator —
same seed and manifest digest yield byte-equivalent trajectories.</h2>

<table class="tb-table">
    <thead>
        <tr>
            <th>Tier</th>
            <th>Bars</th>
            <th>Universe</th>
            <th class="mono">Window</th>
            <th>What it tests</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><strong>T1 · Easy</strong></td>
            <td class="mono">60</td>
            <td>5 aliased assets</td>
            <td class="mono">2020-01-02 → 2020-03-31</td>
            <td>Basic Kelly sizing, short horizon</td>
        </tr>
        <tr>
            <td><strong>T2 · Medium</strong></td>
            <td class="mono">120</td>
            <td>10 aliased assets</td>
            <td class="mono">2020-06-01 → 2020-11-30</td>
            <td>Mid-episode regime navigation, drawdown recovery</td>
        </tr>
        <tr>
            <td><strong>T3 · Hard</strong></td>
            <td class="mono">252</td>
            <td>20 aliased assets</td>
            <td class="mono">2024-01-02 → 2024-12-31</td>
            <td>Full season, post-cutoff anti-memorization stress test</td>
        </tr>
    </tbody>
</table>

<p class="tb-section-lead tb-section-lead-mute">
Bars are <strong>real OHLCV</strong> from yfinance (mega-cap US tickers across
2018→present), baked once by <code>scripts/build_real_dataset.py</code> and
spliced into the catalog. Four anti-memorization layers stack on top:
<em>(1)</em> assets are exposed only as <code>tier_t{1,2,3}_a{NN}</code>
aliases — real symbols never appear in any tool output;
<em>(2)</em> each tier picks an independent random source-window;
<em>(3)</em> the alias↔ticker permutation is per-build-random; and
<em>(4)</em> σ=0.0005 zero-mean Gaussian return noise prevents exact-price
recall. The (window, ticker, permutation, noise-seed) tuple lives in a
private sidecar that <code>progressive_fs</code> never copies into the agent
sandbox. See <code>docs/UI_DESIGN_RATIONALE.md</code> for the full design.
</p>
"""


_LOOP_HTML = """
<div class="tb-section-eyebrow">the daily loop</div>
<h2 class="tb-section-title">observe → model → size → place orders → advance.
Only <em>advance_day</em> moves the clock.</h2>

<svg viewBox="0 0 880 160" style="width: 100%; height: auto; margin-top: 16px;">
    <defs>
        <linearGradient id="loopGrad" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stop-color="#22D3EE"/>
            <stop offset="100%" stop-color="#0E7490"/>
        </linearGradient>
        <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto">
            <path d="M0,0 L10,5 L0,10 z" fill="#22D3EE"/>
        </marker>
    </defs>

    <!-- 5 phase boxes -->
    <g font-family="JetBrains Mono, monospace" font-size="13" fill="#E6E8EB">
        <rect x="20" y="40" width="140" height="80" rx="6" fill="#11161D" stroke="#1F2731"/>
        <text x="90" y="68" text-anchor="middle" fill="#22D3EE" font-size="11" letter-spacing="1.4">OBSERVE</text>
        <text x="90" y="92" text-anchor="middle" fill="#7A8693" font-size="11">view_*  (×6)</text>
        <text x="90" y="108" text-anchor="middle" fill="#7A8693" font-size="11">free, read-only</text>

        <rect x="180" y="40" width="140" height="80" rx="6" fill="#11161D" stroke="#1F2731"/>
        <text x="250" y="68" text-anchor="middle" fill="#22D3EE" font-size="11" letter-spacing="1.4">MODEL</text>
        <text x="250" y="92" text-anchor="middle" fill="#7A8693" font-size="11">sandbox_exec</text>
        <text x="250" y="108" text-anchor="middle" fill="#7A8693" font-size="11">no net, seccomp</text>

        <rect x="340" y="40" width="140" height="80" rx="6" fill="#11161D" stroke="#1F2731"/>
        <text x="410" y="68" text-anchor="middle" fill="#22D3EE" font-size="11" letter-spacing="1.4">SIZE</text>
        <text x="410" y="92" text-anchor="middle" fill="#7A8693" font-size="11">record_decision</text>
        <text x="410" y="108" text-anchor="middle" fill="#7A8693" font-size="11">+ reasoning scan</text>

        <rect x="500" y="40" width="140" height="80" rx="6" fill="#11161D" stroke="#1F2731"/>
        <text x="570" y="68" text-anchor="middle" fill="#22D3EE" font-size="11" letter-spacing="1.4">ORDERS</text>
        <text x="570" y="92" text-anchor="middle" fill="#7A8693" font-size="11">place_order</text>
        <text x="570" y="108" text-anchor="middle" fill="#7A8693" font-size="11">queue, no same-bar</text>

        <rect x="660" y="40" width="200" height="80" rx="6" fill="#1A2029" stroke="#22D3EE" stroke-width="1.5"/>
        <text x="760" y="68" text-anchor="middle" fill="#22D3EE" font-size="11" letter-spacing="1.4">ADVANCE</text>
        <text x="760" y="92" text-anchor="middle" fill="#E6E8EB" font-size="11">next-open fills · settle</text>
        <text x="760" y="108" text-anchor="middle" fill="#E6E8EB" font-size="11">composite reward emitted</text>
    </g>

    <!-- arrows -->
    <g stroke="#22D3EE" stroke-width="1.4" fill="none" marker-end="url(#arr)">
        <line x1="160" y1="80" x2="180" y2="80"/>
        <line x1="320" y1="80" x2="340" y2="80"/>
        <line x1="480" y1="80" x2="500" y2="80"/>
        <line x1="640" y1="80" x2="660" y2="80"/>
    </g>

    <!-- bottom rail label -->
    <text x="440" y="150" text-anchor="middle" font-family="JetBrains Mono, monospace"
          font-size="10" fill="#7A8693" letter-spacing="1.4">
        only ADVANCE_DAY moves t → t+1.  every other tool is free-cost, read-only or queue-only.
    </text>
</svg>
"""


_TOOLS_HTML = """
<div class="tb-section-eyebrow">tool surface — 11 tools</div>
<h2 class="tb-section-title">Six read-only views, three order-book ops,
one time-advancer, one sandboxed Python — plus an optional <code>reasoning</code>
field on <code>record_decision</code> for the rules-clause scanner.</h2>

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
    gr.HTML(_TIERS_HTML)
    gr.HTML(_LOOP_HTML)
    gr.HTML(_TOOLS_HTML)
    gr.HTML(
        '<div class="tb-section-eyebrow">pydantic schemas</div>'
        '<h2 class="tb-section-title">Wire-level contracts. Auto-rendered from '
        '<code>models.py</code> via <code>model_json_schema()</code>.</h2>'
        + _schema_block("TradeAction · action envelope", TradeAction)
        + _schema_block("TradeObservation · per-step response", TradeObservation)
        + _schema_block("TradeState · episode metadata (read via /state)", TradeState),
    )
