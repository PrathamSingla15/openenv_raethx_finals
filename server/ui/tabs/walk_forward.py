"""Defenses tab: six independent look-ahead layers, hack defenses, verifier."""

from __future__ import annotations

import gradio as gr


_TEMPORAL_HTML = """
<div class="tb-section-eyebrow">the temporal contract</div>
<h2 class="tb-section-title">Observation at <em>t</em> sees only data with
<code>available_at ≤ t</code>; reward at <em>t→t+1</em> is computed from
the events ledger up to that boundary, nothing further.</h2>

<svg viewBox="0 0 880 220" style="width: 100%; height: auto; margin-top: 16px;" role="img" aria-label="Temporal contract diagram">
    <defs>
        <marker id="arr2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto">
            <path d="M0,0 L10,5 L0,10 z" fill="#22D3EE"/>
        </marker>
    </defs>

    <line x1="40" y1="180" x2="840" y2="180" stroke="#1F2731" stroke-width="1"/>
    <line x1="40" y1="180" x2="840" y2="180" stroke="#22D3EE" stroke-width="1.5" stroke-dasharray="2 6" opacity="0.45"/>
    <text x="40" y="205" font-family="JetBrains Mono, monospace" font-size="11" fill="#7A8693">t (decision)</text>
    <text x="840" y="205" font-family="JetBrains Mono, monospace" font-size="11" fill="#7A8693" text-anchor="end">t+1 (next session)</text>

    <g font-family="JetBrains Mono, monospace" font-size="11">
        <rect x="60"  y="40" width="120" height="60" rx="4" fill="#11161D" stroke="#22D3EE" stroke-width="1"/>
        <text x="120" y="62" text-anchor="middle" fill="#22D3EE" font-size="10" letter-spacing="1.4">OBSERVE</text>
        <text x="120" y="80" text-anchor="middle" fill="#7A8693" font-size="10">data ≤ t</text>
        <text x="120" y="94" text-anchor="middle" fill="#7A8693" font-size="10">read-only</text>

        <rect x="220" y="40" width="120" height="60" rx="4" fill="#11161D" stroke="#22D3EE" stroke-width="1"/>
        <text x="280" y="62" text-anchor="middle" fill="#22D3EE" font-size="10" letter-spacing="1.4">DECIDE</text>
        <text x="280" y="80" text-anchor="middle" fill="#7A8693" font-size="10">record_decision</text>
        <text x="280" y="94" text-anchor="middle" fill="#7A8693" font-size="10">+ rules scan</text>

        <rect x="380" y="40" width="120" height="60" rx="4" fill="#1A2029" stroke="#22D3EE" stroke-width="1.5"/>
        <text x="440" y="62" text-anchor="middle" fill="#22D3EE" font-size="10" letter-spacing="1.4">SETTLE</text>
        <text x="440" y="80" text-anchor="middle" fill="#E6E8EB" font-size="10">next-open fills</text>
        <text x="440" y="94" text-anchor="middle" fill="#E6E8EB" font-size="10">corp actions</text>

        <rect x="540" y="40" width="120" height="60" rx="4" fill="#1A2029" stroke="#22D3EE" stroke-width="1.5"/>
        <text x="600" y="62" text-anchor="middle" fill="#22D3EE" font-size="10" letter-spacing="1.4">REWARD</text>
        <text x="600" y="80" text-anchor="middle" fill="#E6E8EB" font-size="10">composite breakdown</text>
        <text x="600" y="94" text-anchor="middle" fill="#E6E8EB" font-size="10">7 components × gate</text>

        <rect x="700" y="40" width="120" height="60" rx="4" fill="#11161D" stroke="#22D3EE" stroke-width="1"/>
        <text x="760" y="62" text-anchor="middle" fill="#22D3EE" font-size="10" letter-spacing="1.4">OBSERVE'</text>
        <text x="760" y="80" text-anchor="middle" fill="#7A8693" font-size="10">data ≤ t+1</text>
        <text x="760" y="94" text-anchor="middle" fill="#7A8693" font-size="10">FS refreshed</text>
    </g>

    <g stroke="#22D3EE" stroke-width="1.4" fill="none" marker-end="url(#arr2)">
        <line x1="180" y1="70" x2="220" y2="70"/>
        <line x1="340" y1="70" x2="380" y2="70"/>
        <line x1="500" y1="70" x2="540" y2="70"/>
        <line x1="660" y1="70" x2="700" y2="70"/>
    </g>

    <g font-family="JetBrains Mono, monospace" font-size="10" fill="#7A8693">
        <text x="120" y="135" text-anchor="middle">SQL PIT, date gate</text>
        <text x="120" y="150" text-anchor="middle">progressive FS</text>

        <text x="280" y="135" text-anchor="middle">rules-clause regex</text>
        <text x="280" y="150" text-anchor="middle">decision required</text>

        <text x="440" y="135" text-anchor="middle">slippage cap</text>
        <text x="440" y="150" text-anchor="middle">deterministic order</text>

        <text x="600" y="135" text-anchor="middle">scanners drained</text>
        <text x="600" y="150" text-anchor="middle">violations zero gate</text>

        <text x="760" y="135" text-anchor="middle">refresh_allowed_files</text>
        <text x="760" y="150" text-anchor="middle">advance_day +1</text>
    </g>
</svg>
"""


_LAYERS_HTML = """
<div class="tb-section-eyebrow">six look-ahead defense layers</div>
<h2 class="tb-section-title">LLM weights post-date the backtest window. A single
defense is never enough.</h2>

<div class="tb-layer-stack">
    <div class="tb-layer">
        <div class="tb-layer-num">01</div>
        <div class="tb-layer-content">
            <div class="tb-layer-title">SQL-level PIT filter</div>
            <div class="tb-layer-body">Every DuckDB query filters
                <code>available_at ≤ current_date</code>. The data layer never
                returns a row dated past the agent's clock.</div>
        </div>
    </div>
    <div class="tb-layer">
        <div class="tb-layer-num">02</div>
        <div class="tb-layer-content">
            <div class="tb-layer-title">Tool-level date gating</div>
            <div class="tb-layer-body">Every data-view tool validates its
                <code>as_of_date</code> argument; requests past
                <code>current_date</code> are hard-rejected with a structured
                <code>LookaheadViolation</code>.</div>
        </div>
    </div>
    <div class="tb-layer">
        <div class="tb-layer-num">03</div>
        <div class="tb-layer-content">
            <div class="tb-layer-title">Progressive filesystem view</div>
            <div class="tb-layer-body">Only files with
                <code>available_at ≤ current_date</code> are materialized into the
                sandbox read-only mount. Copy-on-reset, refresh-on-advance.</div>
        </div>
    </div>
    <div class="tb-layer">
        <div class="tb-layer-num">04</div>
        <div class="tb-layer-content">
            <div class="tb-layer-title">Rules-based-strategy clause</div>
            <div class="tb-layer-body">System prompt forbids memorized recalls
                ("I know X happened"). Enforced by regex over the agent's
                <code>reasoning</code> + <code>edge_summary</code>; hits zero the
                compliance gate on the next bar.</div>
        </div>
    </div>
    <div class="tb-layer">
        <div class="tb-layer-num">05</div>
        <div class="tb-layer-content">
            <div class="tb-layer-title">No internet in sandbox</div>
            <div class="tb-layer-body">Docker network_disabled, seccomp profile,
                no-new-privileges, non-root, read-only root. The sandbox is a
                hermetic computation surface, not a data fetcher.</div>
        </div>
    </div>
    <div class="tb-layer">
        <div class="tb-layer-num">06</div>
        <div class="tb-layer-content">
            <div class="tb-layer-title">Aliased + date-shifted data</div>
            <div class="tb-layer-body">Agents see opaque <code>tier_a01</code> ...
                <code>tier_a10</code> IDs against a randomly drawn source window.
                Per-build alias-to-ticker permutation + σ=0.0005 return-noise
                overlay. The underlying real series is never identifiable.</div>
        </div>
    </div>
</div>

<p class="tb-section-lead tb-section-lead-mute">
Each layer is independently bypassable in principle; together they are not.
That is the whole point of a layered defense.
</p>
"""


_AUDIT_HTML = """
<div class="tb-section-eyebrow">per-data-source leakage audit</div>
<h2 class="tb-section-title">Every channel through which future data could
reach the agent — and the gate that closes it.</h2>

<table class="tb-table">
    <thead>
        <tr><th>Source</th><th>Path</th><th>Gate</th></tr>
    </thead>
    <tbody>
        <tr><td>Daily OHLCV</td><td class="mono">data/query.py:get_bars</td><td>SQL <code>available_at ≤ ?</code> + tool-level <code>as_of_date</code> gate</td></tr>
        <tr><td>Universe membership</td><td class="mono">data/query.py:load_universe</td><td>SQL + tool-level gate</td></tr>
        <tr><td>Symbol ↔ asset map</td><td class="mono">data/query.py:get_symbol_to_asset_id</td><td>SQL + tool-level gate</td></tr>
        <tr><td>Calendar / sessions</td><td class="mono">data/query.py:next_session_after</td><td>Strict next; never future</td></tr>
        <tr><td>Corporate actions</td><td class="mono">execution/corporate_actions.py</td><td>Applied at <em>t+1</em>; never visible at decision</td></tr>
        <tr><td><strong>Sandbox filesystem</strong></td><td class="mono">sandbox/workspace.py</td><td><strong>progressive_fs.materialize_allowed_files</strong> on reset; <strong>refresh_allowed_files</strong> on advance_day</td></tr>
        <tr><td>System prompt</td><td class="mono">environment/prompt.py</td><td>Audit: no manifest field carries future data</td></tr>
        <tr><td>Reward inputs</td><td class="mono">rewards/composite.py</td><td>Verifier asserts <code>events_so_far</code> only</td></tr>
        <tr><td>RNG</td><td class="mono">session.py:_rng</td><td>Seeded from manifest digest or <code>reset(seed=…)</code></td></tr>
    </tbody>
</table>
"""


_REWARD_HACK_HTML = """
<div class="tb-section-eyebrow">reward-hacking layer</div>
<h2 class="tb-section-title">A composite reward with a sandboxed code tool gives
the agent multiple knobs to game. Defenses are scoped to the failure mode.</h2>

<ul style="line-height: 1.8; max-width: 64ch;">
    <li><code>scan_forbidden_globals</code> — regex scanner over every
        <code>sandbox_exec</code> stdout/stderr; 12 patterns including
        <code>eval(</code>, <code>__import__</code>, <code>subprocess.</code>,
        <code>socket.</code>, ledger private-attr writes. Hit →
        <code>viol_hack = 1</code>, gate = 0 on the next <code>advance_day</code>.</li>
    <li><code>check_rules_clause</code> — regex scanner over the agent's
        <code>reasoning</code> + <code>edge_summary</code> in
        <code>record_decision</code>. Catches first-person memorized recalls
        ("I recall AAPL crashed in 2020"), ticker-outcome pairs, named regimes.
        Hit → <code>viol_rules = 1</code>, gate = 0.</li>
    <li><strong>Bounded components.</strong> Every <code>cᵢ</code> ∈ [0, 1] by
        construction; no single component can dominate. There is no unbounded
        log-wealth head an agent can chase at the expense of compliance.</li>
    <li><strong>Hard compliance gate.</strong> Multiplicative {0, 1}; rules /
        hack / leverage breach each set <code>g_compliance = 0</code>, zeroing
        the entire bar's reward. There is no additive penalty for cheating to
        out-earn.</li>
    <li><strong>Per-bar turnover cap.</strong> 25 trades. Penny-order exploits
        do not reach the reward function.</li>
</ul>
"""


_VERIFIER_HTML = """
<div class="tb-section-eyebrow">verifier · run after every change</div>
<h2 class="tb-section-title">Six checks. Exit 0 iff every check is PASS or
WARN. Run on each tier as part of CI and at submission.</h2>

<div class="tb-verifier-strip">
    <span class="tb-vcheck">PASS · conformance.observation</span>
    <span class="tb-vcheck">PASS · conformance.action_rejection</span>
    <span class="tb-vcheck">PASS · leak.fs</span>
    <span class="tb-vcheck">PASS · leak.no_future_fit</span>
    <span class="tb-vcheck">PASS · replay</span>
    <span class="tb-vcheck">PASS · determinism</span>
</div>

<details class="tb-schema">
    <summary>$ python -m tradebench.verifier --tier test --seed 42 --max-bars 5</summary>
<pre>TradeBench verifier — tier=test seed=42
  PASS  conformance.observation       all 10 observations valid
  PASS  conformance.action_rejection  malformed actions rejected with structured errors
  PASS  leak.fs[step=10]              inspected 9 files, no future partitions
  PASS  leak.no_future_fit            no preprocessors with fit_through_date > current_date
  PASS  replay                        all components match across 5 steps; max c_alpha delta=0.000e+00
  PASS  determinism                   |ΔV| = 0.000e+00 < 1e-06; 11 events; 10 actions
RESULT: PASS</pre>
</details>
"""


def render() -> None:
    gr.HTML(_TEMPORAL_HTML)
    gr.HTML(_LAYERS_HTML)
    gr.HTML(_AUDIT_HTML)
    gr.HTML(_REWARD_HACK_HTML)
    gr.HTML(_VERIFIER_HTML)
