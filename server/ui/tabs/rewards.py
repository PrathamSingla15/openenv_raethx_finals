"""Reward tab — the 7-component convex composite formula, weights chart,
exploit register, and the live RewardBreakdown dataclass."""

from __future__ import annotations

import gradio as gr


_INTRO_HTML = r"""
<div class="tb-section-eyebrow">composite reward &middot; 7 components in [0, 1]</div>
<h2 class="tb-section-title">
    Per-bar reward is in <span class="tb-num">[0, 1]</span>
    <span class="tb-faded">by construction. Episode score is the mean per-bar
    reward, also in</span> <span class="tb-num">[0, 1]</span>.
</h2>

<p class="tb-section-lead">
    The seven components are each in [0, 1], the seven weights sum to 1.0, and
    the compliance gate is {0, 1}. Cheating zeros the bar's reward outright;
    there is no way for a great Sharpe to earn back a rules-clause hit.
</p>

<div class="tb-formula">
<pre>r_t = (&Sigma;<sub>i</sub> w<sub>i</sub> &middot; c<sub>i,t</sub>) &middot; g_compliance<sub>t</sub>                    # in [0, 1]

components (each in [0, 1]):
  c_alpha       sigmoid(8 &middot; cumulative_log_alpha_vs_bench)        w = 0.40
  c_return      sigmoid(5 &middot; cumulative_log_return)                w = 0.15
  c_drawdown    1 &minus; 2 &middot; min(dd, 0.5)                              w = 0.10
  c_solvency    sigmoid(6 &middot; (V &minus; 0.5&middot;V<sub>0</sub>) / (0.5&middot;V<sub>0</sub>))              w = 0.10
  c_efficiency  exp(&minus;2 &middot; max(0, turnover &minus; 0.10))                 w = 0.10
  c_diversity   1 &minus; clip((HHI &minus; 0.10) / 0.90, 0, 1)               w = 0.05
  c_consistency win &middot; stability + (1 &minus; win) &middot; 0.5                 w = 0.10

compliance gate (multiplicative, not additive):
  g_compliance = (1 &minus; viol_rules) &middot; (1 &minus; viol_hack) &middot; (1 &minus; lev_breach)
  with lev_breach = 1{gross_leverage &gt; 1.0}.   Any single violation zeros the bar.

episode score:
  score_normalized = mean over bars of r_t                        in [0, 1]</pre>
</div>
"""


# Inline SVG bar chart for the convex composite weights. Pure-black canvas,
# matches the editorial palette without Plotly overhead.
_RECIPE_SVG_HTML = """
<figure class="tb-figure">
    <div class="tb-figure-head">
        <div class="tb-figure-title">convex composite weights</div>
        <div class="tb-figure-meta">7 components &middot; sum = 1.00</div>
    </div>
    <div class="tb-figure-body is-dark" style="padding: 32px 24px;">
        <svg viewBox="0 0 880 320" style="width: 100%; height: auto;" role="img" aria-label="Convex composite weights">
            <defs>
                <linearGradient id="rwGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stop-color="#67E8F9"/>
                    <stop offset="100%" stop-color="#0E7490"/>
                </linearGradient>
            </defs>
            <g font-family="IBM Plex Mono, monospace" font-size="11" fill="rgba(255,255,255,0.40)">
                <line x1="60" y1="40" x2="60" y2="240" stroke="rgba(255,255,255,0.10)" stroke-width="1"/>
                <line x1="60" y1="240" x2="860" y2="240" stroke="rgba(255,255,255,0.10)" stroke-width="1"/>
                <text x="48" y="44" text-anchor="end">0.40</text>
                <text x="48" y="124" text-anchor="end">0.20</text>
                <text x="48" y="204" text-anchor="end">0.10</text>
                <text x="48" y="244" text-anchor="end">0.00</text>
            </g>
            <!-- 7 bars: c_alpha=.40, c_return=.15, c_drawdown=.10, c_solvency=.10, c_efficiency=.10, c_diversity=.05, c_consistency=.10 -->
            <g>
                <rect x="100" y="80"  width="80" height="160" fill="url(#rwGrad)"/>
                <text x="140" y="68" text-anchor="middle" font-family="IBM Plex Mono, monospace" font-size="13" fill="#FFFFFF" font-weight="500">0.40</text>
                <text x="140" y="262" text-anchor="middle" font-family="IBM Plex Mono, monospace" font-size="11" fill="rgba(255,255,255,0.66)">c_alpha</text>

                <rect x="200" y="180" width="80" height="60" fill="rgba(34,211,238,0.50)"/>
                <text x="240" y="168" text-anchor="middle" font-family="IBM Plex Mono, monospace" font-size="13" fill="#FFFFFF" font-weight="500">0.15</text>
                <text x="240" y="262" text-anchor="middle" font-family="IBM Plex Mono, monospace" font-size="11" fill="rgba(255,255,255,0.66)">c_return</text>

                <rect x="300" y="200" width="80" height="40" fill="rgba(34,211,238,0.40)"/>
                <text x="340" y="188" text-anchor="middle" font-family="IBM Plex Mono, monospace" font-size="13" fill="#FFFFFF" font-weight="500">0.10</text>
                <text x="340" y="262" text-anchor="middle" font-family="IBM Plex Mono, monospace" font-size="11" fill="rgba(255,255,255,0.66)">c_drawdown</text>

                <rect x="400" y="200" width="80" height="40" fill="rgba(34,211,238,0.40)"/>
                <text x="440" y="188" text-anchor="middle" font-family="IBM Plex Mono, monospace" font-size="13" fill="#FFFFFF" font-weight="500">0.10</text>
                <text x="440" y="262" text-anchor="middle" font-family="IBM Plex Mono, monospace" font-size="11" fill="rgba(255,255,255,0.66)">c_solvency</text>

                <rect x="500" y="200" width="80" height="40" fill="rgba(34,211,238,0.40)"/>
                <text x="540" y="188" text-anchor="middle" font-family="IBM Plex Mono, monospace" font-size="13" fill="#FFFFFF" font-weight="500">0.10</text>
                <text x="540" y="262" text-anchor="middle" font-family="IBM Plex Mono, monospace" font-size="11" fill="rgba(255,255,255,0.66)">c_efficiency</text>

                <rect x="600" y="220" width="80" height="20" fill="rgba(34,211,238,0.30)"/>
                <text x="640" y="208" text-anchor="middle" font-family="IBM Plex Mono, monospace" font-size="13" fill="#FFFFFF" font-weight="500">0.05</text>
                <text x="640" y="262" text-anchor="middle" font-family="IBM Plex Mono, monospace" font-size="11" fill="rgba(255,255,255,0.66)">c_diversity</text>

                <rect x="700" y="200" width="80" height="40" fill="rgba(34,211,238,0.40)"/>
                <text x="740" y="188" text-anchor="middle" font-family="IBM Plex Mono, monospace" font-size="13" fill="#FFFFFF" font-weight="500">0.10</text>
                <text x="740" y="262" text-anchor="middle" font-family="IBM Plex Mono, monospace" font-size="11" fill="rgba(255,255,255,0.66)">c_consistency</text>
            </g>
            <text x="440" y="300" text-anchor="middle" font-family="IBM Plex Mono, monospace" font-size="11" fill="rgba(255,255,255,0.40)" letter-spacing="2">
                weights sum to 1.00 &middot; multiplied by {0,1} compliance gate
            </text>
        </svg>
    </div>
</figure>
"""


_DESIGN_NOTES_HTML = """
<div class="tb-section-eyebrow">three design choices that matter</div>
<h2 class="tb-section-title">
    <span class="tb-faded">Why this reward, and not the obvious one.</span>
</h2>

<div class="tb-three-up">
    <div class="tb-card">
        <div class="tb-card-num">01 / 03</div>
        <div class="tb-card-title">c_alpha is dominant on purpose</div>
        <div class="tb-card-body">A lucky cash-only agent should not beat a thoughtful
            one in a year when markets happen to fall. Weighting alpha-vs-equal-weight-B&amp;H
            at 0.40 forces the policy to actually generate edge, not just survive.</div>
    </div>
    <div class="tb-card">
        <div class="tb-card-num">02 / 03</div>
        <div class="tb-card-title">c_consistency is gated</div>
        <div class="tb-card-body">Above-benchmark agents earn it for low downside-alpha
            volatility; sub-benchmark agents fall back to <code>0.5</code> instead of
            <code>0</code>. That removes the "flatline gets free Sharpe" exploit
            that killed our first reward design.</div>
    </div>
    <div class="tb-card">
        <div class="tb-card-num">03 / 03</div>
        <div class="tb-card-title">Compliance gate is multiplicative</div>
        <div class="tb-card-body">Cheating zeros the bar's reward outright; there is
            no way for a great Sharpe to earn back a rules-clause hit. Bounded
            components keep the optimizer honest about <em>what</em> it is improving.</div>
    </div>
</div>
"""


_EXPLOIT_REGISTER_HTML = """
<div class="tb-section-eyebrow">exploit register &middot; attack classes mapped to defenses</div>
<h2 class="tb-section-title">
    Every plausible <span class="tb-faded">reward-hacking strategy</span>
    mapped to its structural defense.
</h2>

<table class="tb-table">
    <thead>
        <tr><th>#</th><th>Exploit</th><th>Structural defense</th><th>Verifier check</th></tr>
    </thead>
    <tbody>
        <tr><td class="mono">1</td><td>Penny-order churn</td><td>c_efficiency = exp(&minus;2&middot;max(0, turn&minus;0.10)); per-bar 25-trade cap; record_decision required</td><td>replay reward; flag if mean(turnover) &gt;&gt; cap with c_efficiency &asymp; 1</td></tr>
        <tr><td class="mono">2</td><td>Buy-and-hold one stock forever</td><td>c_alpha rewards alpha-vs-EW-B&amp;H, not absolute return; c_diversity penalizes HHI spike</td><td>compare on test (regime flip) vs equal-weight baseline</td></tr>
        <tr><td class="mono">3</td><td>Filesystem ls of future partitions</td><td>Progressive FS materialization on advance_day</td><td>leak.fs after every step</td></tr>
        <tr><td class="mono">4</td><td>Future <code>as_of_date</code> in view tools</td><td>Tool-level date gate; LookaheadViolation &rarr; structured error</td><td>conformance.action_rejection</td></tr>
        <tr><td class="mono">5</td><td>Inflate Sharpe with tiny positions</td><td>c_consistency falls back to 0.5 below B&amp;H; c_alpha still dominates</td><td>flag if mean_position_notional &lt; $100 AND c_consistency &rarr; 1</td></tr>
        <tr><td class="mono">6</td><td>Spread to junk tickers to game HHI</td><td>Curated universe; c_diversity bounded; c_alpha punishes bad allocation</td><td>replay; check &not;(c_diversity &asymp; 1 &and; c_alpha drops)</td></tr>
        <tr><td class="mono">7</td><td>End-of-episode Sharpe spike</td><td>No terminal bonus; mean per-bar reward is the score</td><td>argmax(c_consistency_t) &asymp; uniform across episode</td></tr>
        <tr><td class="mono">8</td><td>Reflective ledger mutation in sandbox</td><td>Ledger lives in env process; sandbox is read-only data + writable work</td><td>scan_forbidden_globals catches private-attr mutation patterns</td></tr>
        <tr><td class="mono">9</td><td>Subprocess / socket sandbox escape</td><td>Docker network_disabled + seccomp + no-new-privileges + non-root + read-only root</td><td>scan_forbidden_globals catches subprocess. / socket.</td></tr>
        <tr><td class="mono">10</td><td>eval / exec / __import__ for env access</td><td>Pattern-match on the obvious dynamic-execution surface</td><td>tests/rewards/test_anti_hack.py covers all 12 patterns</td></tr>
        <tr><td class="mono">11</td><td>Memorized recall ("AAPL crashed in March 2020")</td><td>Rules clause + regex over reasoning + edge_summary; viol_rules &rarr; gate = 0</td><td>tests/rewards/test_anti_hack.py rules-clause cases</td></tr>
        <tr><td class="mono">12</td><td>Game cumulative score by tripping early termination</td><td>Termination-on-violation does not improve cumulative reward</td><td>replay; viol-zeroed episodes total worse than honest completions</td></tr>
    </tbody>
</table>
"""


_DATACLASS_HTML = """
<div class="tb-section-eyebrow">surface &middot; RewardBreakdown</div>
<h2 class="tb-section-title">
    <span class="tb-faded">The dataclass the training loop</span> actually sees.
</h2>

<details class="tb-schema" open>
    <summary>src/tradebench/rewards/composite.py</summary>
<pre>@dataclass(frozen=True)
class RewardBreakdown:
    c_alpha:       float    # [0, 1]  sigmoid(8 &middot; cum_log_alpha_vs_bench)
    c_return:      float    # [0, 1]  sigmoid(5 &middot; cum_log_return)
    c_drawdown:    float    # [0, 1]  1 &minus; 2 &middot; min(dd, 0.5)
    c_solvency:    float    # [0, 1]  sigmoid(6 &middot; (V &minus; 0.5&middot;V0) / (0.5&middot;V0))
    c_efficiency:  float    # [0, 1]  exp(&minus;2 &middot; max(0, turnover &minus; 0.10))
    c_diversity:   float    # [0, 1]  1 &minus; clip((HHI &minus; 0.10) / 0.90, 0, 1)
    c_consistency: float    # [0, 1]  win &middot; stability + (1 &minus; win) &middot; 0.5
    g_compliance:  int      # {0, 1}  zeros the bar on any violation

    weights: ClassVar[Mapping[str, float]] = {
        "c_alpha":       0.40,
        "c_return":      0.15,
        "c_drawdown":    0.10,
        "c_solvency":    0.10,
        "c_efficiency":  0.10,
        "c_diversity":   0.05,
        "c_consistency": 0.10,
    }

    def total(self) -> float:
        composite = sum(self.weights[k] * getattr(self, k) for k in self.weights)
        return composite * self.g_compliance</pre>
</details>
"""


def render() -> None:
    gr.HTML(_INTRO_HTML)
    gr.HTML(_RECIPE_SVG_HTML)
    gr.HTML(_DESIGN_NOTES_HTML)
    gr.HTML(_EXPLOIT_REGISTER_HTML)
    gr.HTML(_DATACLASS_HTML)
