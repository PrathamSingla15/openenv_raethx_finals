"""Overview tab — capability gap, walk-forward hero, design pillars."""

from __future__ import annotations

import gradio as gr


_HERO_HTML = """
<div class="tb-timeline">
    <div class="tb-section-eyebrow">walk-forward window</div>
    <div class="tb-timeline-rail">
        <div class="tb-timeline-seg t1">T1 · 60 bars</div>
        <div class="tb-timeline-seg embargo"></div>
        <div class="tb-timeline-seg t2">T2 · 120 bars</div>
        <div class="tb-timeline-seg embargo"></div>
        <div class="tb-timeline-seg t3">T3 · 252 bars</div>
    </div>
    <div class="tb-timeline-axis">
        <div>2020-01 → 03</div>
        <div>—</div>
        <div>2020-06 → 11</div>
        <div>—</div>
        <div>2024-01 → 12</div>
    </div>
</div>

<div class="tb-metric-row">
    <div class="tb-metric">
        <div class="tb-metric-label">composite reward</div>
        <div class="tb-metric-value">7-vec</div>
        <div class="tb-metric-sub">log-wealth + 6 bounded regularizers</div>
    </div>
    <div class="tb-metric">
        <div class="tb-metric-label">leak defenses</div>
        <div class="tb-metric-value">5</div>
        <div class="tb-metric-sub">PIT · date-gate · progressive FS · rules clause · post-cutoff tier</div>
    </div>
    <div class="tb-metric">
        <div class="tb-metric-label">verifier checks</div>
        <div class="tb-metric-value">6</div>
        <div class="tb-metric-sub">conformance · leak · replay · determinism · termination · edge</div>
    </div>
</div>
"""

_GAP_MD = """
<div class="tb-section-eyebrow">the capability gap</div>

## Frontier LLMs fail at long-horizon, non-stationary decision-making under uncertainty.

Today's models can reason about a single trading decision in isolation, but
break down across hundreds of sequential choices where past actions reshape
future state, correlations drift across regimes, and weight memorization gives
a false sense of edge. **TradeBench is the equities-trading instance of that
gap, instrumented for training, not just scoring.**

- 11 tools, 60 / 120 / 252-bar episodes, point-in-time data with no peeking.
- Dense Kelly-optimal log-wealth reward decomposed into seven components a
  GRPO trainer can attribute credit across.
- Six independent look-ahead defenses stacked so the agent must derive
  strategy from the data it sees, not retrieve it from training memory.
- A six-check verifier that has to PASS for any submission claim to hold.
"""


_DESIGN_MD = """
<div class="tb-section-eyebrow">design pillars</div>

## Three commitments that distinguish this env from existing finance-LLM benchmarks.

### 1. Dense Kelly-optimal reward, decomposed for credit assignment
The primary signal is `log(V_{t+1}/V_t)` — Kelly-optimal, symmetric around ruin,
the canonical objective for long-run geometric growth. Six bounded regularizers
(Sharpe bonus, drawdown, turnover, concentration, rules-compliance, anti-hack)
shape behavior without dominating the wealth signal. GRPO consumes the
seven components as an independent reward vector — credit attribution per
component, not just per scalar.

### 2. Layered look-ahead defenses
SQL-level point-in-time filter at the DuckDB layer, tool-level date-gating that
hard-rejects future `as_of_date`, progressive filesystem materialization that
copies only past-dated files into the sandbox mount, a rules-based-strategy
clause appended to every prompt and verified by regex on agent reasoning, and
a post-cutoff T3 test tier. **Each layer is independently bypassable in
principle; together they are not.**

### 3. Verifier-asserted, not verifier-claimed
Every claim above lives in `python -m tradebench.verifier --tier {t1,t2,t3}` —
six checks that exit non-zero on any failure. Reward replay equality, sandbox
filesystem audit, scalar-tolerance determinism, observation Pydantic round-trip,
malformed-action rejection, end-of-window termination. **PASS** on all three
tiers as of submission.
"""


def render() -> None:
    gr.HTML(_HERO_HTML)
    gr.Markdown(_GAP_MD)
    gr.Markdown(_DESIGN_MD)
