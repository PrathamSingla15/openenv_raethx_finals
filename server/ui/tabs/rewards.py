"""Reward tab: the 7-component convex composite formula, weights chart,
exploit register, and the live RewardBreakdown dataclass."""

from __future__ import annotations

import gradio as gr

from .. import plots


_INTRO_HTML = r"""
<div class="tb-section-eyebrow">composite reward · 7 components in [0, 1]</div>
<h2 class="tb-section-title">Per-bar reward is in [0, 1] by construction. Episode
score is the mean per-bar reward, also in [0, 1].</h2>

<p class="tb-section-lead">
    The seven components are each in [0, 1], the seven weights sum to 1.0, and
    the compliance gate is {0, 1}. Cheating zeros the bar's reward outright;
    there is no way for a great Sharpe to earn back a rules-clause hit.
</p>

<div class="tb-formula">
<pre>r_t = (Σᵢ wᵢ · cᵢ_t) · g_compliance_t                    # in [0, 1]

components (each in [0, 1]):
  c_alpha       sigmoid(8 · cumulative_log_alpha_vs_bench)        w = 0.40
  c_return      sigmoid(5 · cumulative_log_return)                w = 0.15
  c_drawdown    1 − 2 · min(dd, 0.5)                              w = 0.10
  c_solvency    sigmoid(6 · (V − 0.5·V₀) / (0.5·V₀))              w = 0.10
  c_efficiency  exp(−2 · max(0, turnover − 0.10))                 w = 0.10
  c_diversity   1 − clip((HHI − 0.10) / 0.90, 0, 1)               w = 0.05
  c_consistency win · stability + (1 − win) · 0.5                 w = 0.10

compliance gate (multiplicative, not additive):
  g_compliance = (1 − viol_rules) · (1 − viol_hack) · (1 − lev_breach)
  with lev_breach = 1{gross_leverage > 1.0}.   Any single violation zeros the bar.

episode score:
  score_normalized = mean over bars of r_t                        in [0, 1]</pre>
</div>
"""


_DESIGN_NOTES_HTML = """
<div class="tb-section-eyebrow">three design choices that matter</div>

<div class="tb-three-up">
    <div class="tb-card">
        <div class="tb-card-num">01</div>
        <div class="tb-card-title">c_alpha is dominant on purpose</div>
        <div class="tb-card-body">A lucky cash-only agent should not beat a thoughtful
            one in a year when markets happen to fall. Weighting alpha-vs-equal-weight-B&H
            at 0.40 forces the policy to actually generate edge, not just survive.</div>
    </div>
    <div class="tb-card">
        <div class="tb-card-num">02</div>
        <div class="tb-card-title">c_consistency is gated</div>
        <div class="tb-card-body">Above-benchmark agents earn it for low downside-alpha
            volatility; sub-benchmark agents fall back to <code>0.5</code> instead of
            <code>0</code>. That removes the "flatline gets free Sharpe" exploit
            that killed our first reward design.</div>
    </div>
    <div class="tb-card">
        <div class="tb-card-num">03</div>
        <div class="tb-card-title">Compliance gate is multiplicative</div>
        <div class="tb-card-body">Cheating zeros the bar's reward outright; there is
            no way for a great Sharpe to earn back a rules-clause hit. Bounded
            components keep the optimizer honest about <em>what</em> it is improving.</div>
    </div>
</div>
"""


_EXPLOIT_REGISTER_HTML = """
<div class="tb-section-eyebrow">exploit register · attack classes mapped to defenses</div>
<h2 class="tb-section-title">Every plausible reward-hacking strategy mapped to its
structural defense and the verifier check that proves it.</h2>

<table class="tb-table">
    <thead>
        <tr><th>#</th><th>Exploit</th><th>Structural defense</th><th>Verifier check</th></tr>
    </thead>
    <tbody>
        <tr><td class="mono">1</td><td>Penny-order churn</td><td>c_efficiency = exp(−2·max(0, turn−0.10)); per-bar 25-trade cap; record_decision required</td><td>replay reward; flag if mean(turnover) ≫ cap with c_efficiency ≈ 1</td></tr>
        <tr><td class="mono">2</td><td>Buy-and-hold one stock forever</td><td>c_alpha rewards alpha-vs-EW-B&amp;H, not absolute return; c_diversity penalizes HHI spike</td><td>compare on test (regime flip) vs equal-weight baseline</td></tr>
        <tr><td class="mono">3</td><td>Filesystem ls of future partitions</td><td>Progressive FS materialization on advance_day</td><td>leak.fs after every step</td></tr>
        <tr><td class="mono">4</td><td>Future <code>as_of_date</code> in view tools</td><td>Tool-level date gate; LookaheadViolation → structured error</td><td>conformance.action_rejection</td></tr>
        <tr><td class="mono">5</td><td>Inflate Sharpe with tiny positions</td><td>c_consistency falls back to 0.5 below B&amp;H; c_alpha still dominates</td><td>flag if mean_position_notional &lt; $100 AND c_consistency → 1</td></tr>
        <tr><td class="mono">6</td><td>Spread to junk tickers to game HHI</td><td>Curated universe; c_diversity bounded; c_alpha punishes bad allocation</td><td>replay; check ¬(c_diversity ≈ 1 ∧ c_alpha drops)</td></tr>
        <tr><td class="mono">7</td><td>End-of-episode Sharpe spike</td><td>No terminal bonus; mean per-bar reward is the score</td><td>argmax(c_consistency_t) ≈ uniform across episode</td></tr>
        <tr><td class="mono">8</td><td>Reflective ledger mutation in sandbox</td><td>Ledger lives in env process; sandbox is read-only data + writable work</td><td>scan_forbidden_globals catches private-attr mutation patterns</td></tr>
        <tr><td class="mono">9</td><td>Subprocess / socket sandbox escape</td><td>Docker network_disabled + seccomp + no-new-privileges + non-root + read-only root</td><td>scan_forbidden_globals catches subprocess. / socket.</td></tr>
        <tr><td class="mono">10</td><td>eval / exec / __import__ for env access</td><td>Pattern-match on the obvious dynamic-execution surface</td><td>tests/rewards/test_anti_hack.py covers all 12 patterns</td></tr>
        <tr><td class="mono">11</td><td>Memorized recall ("AAPL crashed in March 2020")</td><td>Rules clause + regex over reasoning + edge_summary; viol_rules → gate = 0</td><td>tests/rewards/test_anti_hack.py rules-clause cases</td></tr>
        <tr><td class="mono">12</td><td>Game cumulative score by tripping early termination</td><td>Termination-on-violation does not improve cumulative reward</td><td>replay; viol-zeroed episodes total worse than honest completions</td></tr>
    </tbody>
</table>
"""


_DATACLASS_HTML = """
<div class="tb-section-eyebrow">surface · RewardBreakdown</div>
<h2 class="tb-section-title">The dataclass the training loop actually sees.</h2>

<details class="tb-schema" open>
    <summary>src/tradebench/rewards/composite.py</summary>
<pre>@dataclass(frozen=True)
class RewardBreakdown:
    c_alpha:       float    # [0, 1]  sigmoid(8 · cum_log_alpha_vs_bench)
    c_return:      float    # [0, 1]  sigmoid(5 · cum_log_return)
    c_drawdown:    float    # [0, 1]  1 − 2 · min(dd, 0.5)
    c_solvency:    float    # [0, 1]  sigmoid(6 · (V − 0.5·V0) / (0.5·V0))
    c_efficiency:  float    # [0, 1]  exp(−2 · max(0, turnover − 0.10))
    c_diversity:   float    # [0, 1]  1 − clip((HHI − 0.10) / 0.90, 0, 1)
    c_consistency: float    # [0, 1]  win · stability + (1 − win) · 0.5
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
    gr.Plot(value=plots.reward_recipe(), show_label=False)
    gr.HTML(_DESIGN_NOTES_HTML)
    gr.HTML(_EXPLOIT_REGISTER_HTML)
    gr.HTML(_DATACLASS_HTML)
