"""Rewards tab — animated component bars, worked example, exploit register."""

from __future__ import annotations

import gradio as gr


_INTRO_HTML = r"""
<div class="tb-section-eyebrow">composite reward · 7 components</div>
<h2 class="tb-section-title">Primary log-wealth carries the objective; six bounded
regularizers shape behavior and deny exploits.</h2>

<p class="tb-section-lead">
Every <code>advance_day</code> emits a <code>RewardBreakdown</code> dataclass with seven
floats. The training scalar is
<code>r_wealth + 0.5 · Σ secondary</code>; the per-component vector flows
unweighted to GRPO so credit attribution is preserved.
</p>
"""


def _bar(name: str, label: str, value: float, lo: float, hi: float, hint: str = "") -> str:
    """Animated horizontal bar from value within [lo, hi].

    Negative bars grow from the right edge of the zero gutter; positive
    bars from the left. Heuristic class flips color (long / short / accent).
    """

    span = max(abs(lo), abs(hi)) or 1.0
    pct = min(100.0, abs(value) / span * 100.0)
    direction = "neg" if value < 0 else ("pos" if value > 0 and ("wealth" in name or name == "r_sharpe_bonus") else "accent")
    if value == 0.0:
        pct = 0.5
        direction = "accent"
    bar_left = "0" if value >= 0 else f"{50 - pct/2:.2f}%"
    return f"""
<div class="tb-reward-row">
    <div class="tb-reward-name">{name}</div>
    <div class="tb-reward-bar-track">
        <div class="tb-reward-bar-fill {direction}"
             style="left: {bar_left}; width: {pct:.2f}%;"></div>
    </div>
    <div class="tb-reward-value">{value:+.4f}</div>
    <div class="tb-reward-bound">{label}</div>
</div>"""


_WORKED_EXAMPLE_HTML = (
    """
<div class="tb-section-eyebrow">worked example · t1 day 12 of 60</div>
<h2 class="tb-section-title">A single bar, decomposed.</h2>

<p class="tb-section-lead tb-section-lead-mute">
V<sub>before</sub> = $103,200 · V<sub>after</sub> = $103,950 · gross
notional this bar = $48,000 · running HWM = $104,500 · 20-bar Sharpe = 1.6 ·
end-of-day HHI = 0.42 · no rules / hack hits.
</p>

<div class="tb-reward-table">
"""
    + _bar("r_wealth",         "[unbounded]",     0.00724, -0.05, 0.05,  "log(103950/103200)")
    + _bar("r_sharpe_bonus",   "[0, 0.20]",       0.20,    -0.20, 0.20,  "clip(1.6/2, 0, 0.2)")
    + _bar("r_drawdown",       "[-0.30, 0]",     -0.0000277, -0.30, 0.0, "-(550/104500)^2")
    + _bar("r_turnover",       "[-0.10, 0]",      0.0,     -0.10, 0.0,  "0.465 < 0.5 threshold")
    + _bar("r_concentration",  "[-0.10, 0]",      0.0,     -0.10, 0.0,  "0.42 < 0.5 threshold")
    + _bar("r_rules",          "{-1, 0}",         0.0,     -1.0,  0.0,  "")
    + _bar("r_hack",           "{-1, 0}",         0.0,     -1.0,  0.0,  "")
    + """
</div>

<p class="tb-section-lead">
Telemetry scalar:
<code>r_total = 0.00724 + 0.5 × (0.20 + ...) ≈ +0.10723</code>.
</p>
<p class="tb-section-lead tb-section-lead-mute">
GRPO consumes the seven components as an independent reward vector; group-relative
advantages are computed per signal and combined inside TRL. The scalar above is
for diagnostic plots only.
</p>
"""
)


_EXPLOIT_REGISTER_HTML = """
<div class="tb-section-eyebrow">exploit register · 13 attack classes</div>
<h2 class="tb-section-title">Every plausible reward-hacking strategy mapped to
its structural defense and verifier check.</h2>

<table class="tb-table">
    <thead>
        <tr><th>#</th><th>Exploit</th><th>Structural defense</th><th>Verifier check</th></tr>
    </thead>
    <tbody>
        <tr><td class="mono">1</td><td>Penny-order churn</td><td>r_turnover > 0.5; per-tier order caps; record_decision required</td><td>replay reward; flag if mean(turnover) ≫ cap with positive r_turnover</td></tr>
        <tr><td class="mono">2</td><td>Buy-and-hold one stock forever</td><td>r_wealth rewards growth not preservation; cash baseline establishes floor</td><td>compare on T2 (regime flip) vs cash baseline</td></tr>
        <tr><td class="mono">3</td><td>Filesystem ls of future partitions</td><td>Progressive FS materialization on advance_day</td><td>leak.fs after every step</td></tr>
        <tr><td class="mono">4</td><td>Future <code>as_of_date</code> in view tools</td><td>Tool-level date gate; LookaheadViolation → structured error</td><td>conformance.action_rejection</td></tr>
        <tr><td class="mono">5</td><td>Inflate Sharpe with tiny positions</td><td>r_sharpe_bonus clipped at +0.2; r_wealth still dominates GRPO advantage</td><td>flag if mean_position_notional &lt; $100 AND r_sharpe → 0.2</td></tr>
        <tr><td class="mono">6</td><td>Spread to junk tickers to game HHI</td><td>Curated universe; HHI penalty bounded; r_wealth punishes bad allocation</td><td>replay; check ⌐(r_concentration ≈ 0 ∧ r_wealth &lt; 0)</td></tr>
        <tr><td class="mono">7</td><td>End-of-episode Sharpe spike</td><td>No terminal bonus; bar-by-bar wealth signal</td><td>argmax(r_sharpe_t over t) ≈ uniform across episode</td></tr>
        <tr><td class="mono">8</td><td>Reflective ledger mutation in sandbox</td><td>Ledger lives in env process; sandbox is read-only data + writable work</td><td>scan_forbidden_globals catches private-attr mutation patterns</td></tr>
        <tr><td class="mono">9</td><td>Subprocess / socket sandbox escape</td><td>Docker network_disabled, seccomp, no-new-privileges, non-root, read-only root</td><td>scan_forbidden_globals catches subprocess. / socket.</td></tr>
        <tr><td class="mono">10</td><td>eval / exec / __import__ for env access</td><td>Pattern-match on the obvious dynamic-execution surface</td><td>tests/rewards/test_anti_hack.py covers all 12 patterns</td></tr>
        <tr><td class="mono">11</td><td>Memorized recall ("AAPL crashed in March 2020")</td><td>Rules clause; regex over reasoning + edge_summary; r_rules = -1.0</td><td>tests/rewards/test_anti_hack.py rules-clause cases</td></tr>
        <tr><td class="mono">12</td><td>Fit-on-future feature stats</td><td>Verifier asserts cached preprocessor.fit_through_date ≤ current_date</td><td>leak.no_future_fit (forward-compat)</td></tr>
        <tr><td class="mono">13</td><td>Game cumulative score by tripping early termination</td><td>Termination-on-violation does not improve cumulative reward</td><td>replay; r_rules = -1 episodes total worse than honest completions</td></tr>
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
    r_wealth: float          # log(V_t1/V_t0)             — unbounded, Kelly-optimal
    r_sharpe_bonus: float    # [0.0, 0.2]                 — rolling Sharpe / 2, clipped
    r_drawdown: float        # [-0.3, 0.0]                — -(new_dd_pct)² below HWM
    r_turnover: float        # [-0.1, 0.0]                — -0.2 · (turnover - 0.5)⁺
    r_concentration: float   # [-0.1, 0.0]                — -0.2 · (HHI - 0.5)⁺
    r_rules: float           # {-1.0, 0.0}                — rules-clause regex hit
    r_hack: float            # {-1.0, 0.0}                — forbidden-global in sandbox

    def total(self) -> float:
        secondary = (self.r_sharpe_bonus + self.r_drawdown + self.r_turnover
                     + self.r_concentration + self.r_rules + self.r_hack)
        return self.r_wealth + 0.5 * secondary</pre>
</details>
"""


def render() -> None:
    gr.HTML(_INTRO_HTML)
    gr.HTML(_WORKED_EXAMPLE_HTML)
    gr.HTML(_EXPLOIT_REGISTER_HTML)
    gr.HTML(_DATACLASS_HTML)
