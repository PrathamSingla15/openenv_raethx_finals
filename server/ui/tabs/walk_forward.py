"""Walk-Forward & Leak Prevention tab.

Frames the temporal contract first, then the defenses that enforce it,
then the verifier output that confirms.
"""

from __future__ import annotations

import gradio as gr


_TEMPORAL_HTML = """
<div class="tb-section-eyebrow">the temporal contract</div>
<h2 class="tb-section-title">Observation at <em>t</em> sees only data with
<code>available_at ≤ t</code>; reward at <em>t→t+1</em> is computed from
the events ledger up to that boundary, nothing further.</h2>

<svg viewBox="0 0 880 220" style="width: 100%; height: auto; margin-top: 16px;">
    <defs>
        <marker id="arr2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto">
            <path d="M0,0 L10,5 L0,10 z" fill="#22D3EE"/>
        </marker>
    </defs>

    <!-- horizontal time axis -->
    <line x1="40" y1="180" x2="840" y2="180" stroke="#1F2731" stroke-width="1"/>
    <line x1="40" y1="180" x2="840" y2="180" stroke="#22D3EE" stroke-width="1.5" stroke-dasharray="2 6" opacity="0.45"/>
    <text x="40" y="205" font-family="JetBrains Mono, monospace" font-size="11" fill="#7A8693">t (decision)</text>
    <text x="840" y="205" font-family="JetBrains Mono, monospace" font-size="11" fill="#7A8693" text-anchor="end">t+1 (next session)</text>

    <!-- 5 phase pills -->
    <g font-family="JetBrains Mono, monospace" font-size="11">
        <rect x="60" y="40" width="120" height="60" rx="4" fill="#11161D" stroke="#22D3EE" stroke-width="1"/>
        <text x="120" y="62" text-anchor="middle" fill="#22D3EE" font-size="10" letter-spacing="1.4">OBSERVE</text>
        <text x="120" y="80" text-anchor="middle" fill="#7A8693" font-size="10">data ≤ t</text>
        <text x="120" y="94" text-anchor="middle" fill="#7A8693" font-size="10">read-only</text>

        <rect x="220" y="40" width="120" height="60" rx="4" fill="#11161D" stroke="#22D3EE" stroke-width="1"/>
        <text x="280" y="62" text-anchor="middle" fill="#22D3EE" font-size="10" letter-spacing="1.4">ACT</text>
        <text x="280" y="80" text-anchor="middle" fill="#7A8693" font-size="10">queue orders</text>
        <text x="280" y="94" text-anchor="middle" fill="#7A8693" font-size="10">no fills yet</text>

        <rect x="380" y="40" width="120" height="60" rx="4" fill="#1A2029" stroke="#22D3EE" stroke-width="1.5"/>
        <text x="440" y="62" text-anchor="middle" fill="#22D3EE" font-size="10" letter-spacing="1.4">SETTLE</text>
        <text x="440" y="80" text-anchor="middle" fill="#E6E8EB" font-size="10">next-open fills</text>
        <text x="440" y="94" text-anchor="middle" fill="#E6E8EB" font-size="10">corp actions</text>

        <rect x="540" y="40" width="120" height="60" rx="4" fill="#1A2029" stroke="#22D3EE" stroke-width="1.5"/>
        <text x="600" y="62" text-anchor="middle" fill="#22D3EE" font-size="10" letter-spacing="1.4">REWARD</text>
        <text x="600" y="80" text-anchor="middle" fill="#E6E8EB" font-size="10">composite breakdown</text>
        <text x="600" y="94" text-anchor="middle" fill="#E6E8EB" font-size="10">7 components</text>

        <rect x="700" y="40" width="120" height="60" rx="4" fill="#11161D" stroke="#22D3EE" stroke-width="1"/>
        <text x="760" y="62" text-anchor="middle" fill="#22D3EE" font-size="10" letter-spacing="1.4">OBSERVE'</text>
        <text x="760" y="80" text-anchor="middle" fill="#7A8693" font-size="10">data ≤ t+1</text>
        <text x="760" y="94" text-anchor="middle" fill="#7A8693" font-size="10">FS refreshed</text>
    </g>

    <!-- arrows between pills -->
    <g stroke="#22D3EE" stroke-width="1.4" fill="none" marker-end="url(#arr2)">
        <line x1="180" y1="70" x2="220" y2="70"/>
        <line x1="340" y1="70" x2="380" y2="70"/>
        <line x1="500" y1="70" x2="540" y2="70"/>
        <line x1="660" y1="70" x2="700" y2="70"/>
    </g>

    <!-- contract callouts -->
    <g font-family="JetBrains Mono, monospace" font-size="10" fill="#7A8693">
        <text x="120" y="135" text-anchor="middle">SQL PIT, date gate</text>
        <text x="120" y="150" text-anchor="middle">progressive FS</text>

        <text x="280" y="135" text-anchor="middle">turnover cap</text>
        <text x="280" y="150" text-anchor="middle">decision_required</text>

        <text x="440" y="135" text-anchor="middle">slippage cap 50bps</text>
        <text x="440" y="150" text-anchor="middle">deterministic order</text>

        <text x="600" y="135" text-anchor="middle">scanners drained:</text>
        <text x="600" y="150" text-anchor="middle">r_rules / r_hack</text>

        <text x="760" y="135" text-anchor="middle">refresh_allowed_files</text>
        <text x="760" y="150" text-anchor="middle">advance_day +1</text>
    </g>
</svg>
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
        <tr><td>Cached preprocessors</td><td class="mono">(none today)</td><td>Forward-compat: <code>fit_through_date ≤ current_date</code></td></tr>
        <tr><td>RNG</td><td class="mono">session.py:_rng</td><td>Seeded from manifest digest or <code>reset(seed=…)</code></td></tr>
    </tbody>
</table>
"""


_REWARD_HACK_HTML = """
<div class="tb-section-eyebrow">reward-hacking layer</div>
<h2 class="tb-section-title">A composite reward with a sandboxed code tool gives
the agent multiple knobs to game. Defenses are scoped to the failure mode.</h2>

<ul style="line-height: 1.8; max-width: 62ch;">
    <li><code>scan_forbidden_globals</code> — regex scanner over every
        <code>sandbox_exec</code> stdout/stderr; 12 patterns including
        <code>eval(</code>, <code>__import__</code>, <code>subprocess.</code>,
        <code>socket.</code>, ledger private-attr writes. Hit → <code>r_hack = -1.0</code>
        on the next <code>advance_day</code>.</li>
    <li><code>check_rules_clause</code> — regex scanner over the agent's
        <code>reasoning</code> and <code>edge_summary</code> in
        <code>record_decision</code>. Catches first-person memorized recalls
        ("I recall AAPL crashed in 2020"), ticker-outcome pairs, and named
        regimes. Hit → <code>r_rules = -1.0</code>.</li>
    <li><strong>Bounded secondary rewards</strong> — no single regularizer can
        exceed ±0.3 per bar (rules / hack are ±1.0 to dwarf any plausible
        gain). The math forces the policy to learn the underlying skill
        instead of finding a soft target.</li>
    <li><strong>Per-tier turnover caps</strong> — T1: 10 trades / day, T2: 25,
        T3: 50. Penny-order exploits don't reach the reward function.</li>
    <li><strong>Periodic generation inspection</strong> — the training loop
        samples 1-in-N rollouts to dump generations for visual review
        (per the hackathon §7.14 guidance: don't watch a single scalar).</li>
</ul>
"""


_VERIFIER_HTML = """
<div class="tb-section-eyebrow">verifier · run after every change</div>
<h2 class="tb-section-title">Six checks. Exit 0 iff every check is PASS or
WARN. Run on T1, T2, T3 as part of CI and at submission.</h2>

<div class="tb-verifier-strip">
    <span class="tb-vcheck">PASS · conformance.observation</span>
    <span class="tb-vcheck">PASS · conformance.action_rejection</span>
    <span class="tb-vcheck">PASS · leak.fs</span>
    <span class="tb-vcheck">PASS · leak.no_future_fit</span>
    <span class="tb-vcheck">PASS · replay</span>
    <span class="tb-vcheck">PASS · determinism</span>
</div>

<details class="tb-schema">
    <summary>$ python -m tradebench.verifier --tier t1 --seed 42 --max-bars 5</summary>
<pre>TradeBench verifier — tier=t1 seed=42
  PASS  conformance.observation       all 10 observations valid
  PASS  conformance.action_rejection  malformed actions rejected with structured errors
  PASS  leak.fs[step=10]              inspected 9 files, no future partitions
  PASS  leak.no_future_fit            no preprocessors with fit_through_date > current_date
  PASS  replay                        all components match across 5 steps; max r_wealth delta=0.000e+00
  PASS  determinism                   |ΔV| = 0.000e+00 < 1e-06; 11 events; 10 actions
RESULT: PASS</pre>
</details>
"""


def render() -> None:
    gr.HTML(_TEMPORAL_HTML)
    gr.HTML(_AUDIT_HTML)
    gr.HTML(_REWARD_HACK_HTML)
    gr.HTML(_VERIFIER_HTML)
