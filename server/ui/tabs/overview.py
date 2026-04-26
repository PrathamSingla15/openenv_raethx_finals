"""Overview tab: headline-results-first hero with the trajectory chart."""

from __future__ import annotations

import gradio as gr

from .. import plots


_HERO_HTML = """
<div class="tb-hero">
    <div class="tb-hero-stack">
        <div class="tb-section-eyebrow">5 reflections · 0 gradient updates · 119-bar test episode</div>
        <h2 class="tb-hero-title">
            Frozen <span class="tb-accent">Qwen3-32B</span> climbed
            <span class="tb-accent">+4.3 pp</span> on score and
            <span class="tb-accent">3.6×</span> on ROI by
            <span class="tb-accent">rewriting its own prompt</span>.
        </h2>
        <p class="tb-hero-lead">
            TradeBench is a long-horizon equities-trading environment built so an LLM agent
            can actually <em>learn</em> on it. The reward is a 7-component convex composite
            in [0, 1] gated by a binary compliance check; the data is real OHLCV behind a
            four-layer anti-memorization stack; and a single 119-bar test rollout produces
            300-500 LLM calls, dense per-bar reward, and a deterministic grader.
            We took two open-weights base models, ran a GEPA-style reflection loop that
            never touches a weight, and got monotone score climbs on both.
        </p>
    </div>

    <div class="tb-hero-stats">
        <div class="tb-statcard">
            <div class="tb-statcard-eyebrow">Qwen3-32B · Groq</div>
            <div class="tb-statcard-row">
                <div class="tb-statcard-key">score_normalized</div>
                <div class="tb-statcard-arrow">0.6155 → <strong>0.6584</strong></div>
            </div>
            <div class="tb-statcard-row">
                <div class="tb-statcard-key">ROI</div>
                <div class="tb-statcard-arrow">+2.78% → <strong>+9.89%</strong></div>
            </div>
            <div class="tb-statcard-row">
                <div class="tb-statcard-key">place_order</div>
                <div class="tb-statcard-arrow">4 → <strong>12</strong></div>
            </div>
            <div class="tb-statcard-foot">5 iterations · monotone</div>
        </div>
        <div class="tb-statcard">
            <div class="tb-statcard-eyebrow">GLM-5.1 · Together</div>
            <div class="tb-statcard-row">
                <div class="tb-statcard-key">score_normalized</div>
                <div class="tb-statcard-arrow">0.6243 → <strong>0.6453</strong></div>
            </div>
            <div class="tb-statcard-row">
                <div class="tb-statcard-key">ROI</div>
                <div class="tb-statcard-arrow">+2.83% → <strong>+4.69%</strong></div>
            </div>
            <div class="tb-statcard-row">
                <div class="tb-statcard-key">place_order</div>
                <div class="tb-statcard-arrow">2 → <strong>6</strong></div>
            </div>
            <div class="tb-statcard-foot">3 iterations · climb · dip · recover</div>
        </div>
    </div>
</div>
"""


_PILLARS_MD = """
<div class="tb-section-eyebrow">design pillars</div>

## Three commitments that distinguish this environment.

### 1. Per-bar reward in [0, 1] by construction
A convex combination of seven trader-recognizable components (alpha vs equal-weight
buy-and-hold at 0.40 weight, cumulative return, drawdown, solvency, turnover
efficiency, concentration, downside-volatility-of-bar-alpha) multiplied by a
{0, 1} compliance gate. The episode score is the mean per-bar reward, also in
[0, 1]. No unbounded log-wealth head, no flatline-gets-free-Sharpe exploit, no
additive penalty that a great quarter could earn back.

### 2. Long-horizon planning spine, server-enforced
A `record_decision` (regime label, edge summary, intended exposure, top
convictions, uncertainty) is mandatory before every `advance_day`. The agent
literally cannot tick the clock without committing a thesis on the record.
Every advance leaves a paper trail the reward function can audit.

### 3. Anti-memorization is structural, not aspirational
Aliased universe (`tier_a01` ... `tier_a10`, real ticker symbols never appear
in any tool output) layered with a randomly drawn source window from the broad
[2018, today] pool, per-build alias-to-ticker permutation, and σ=0.0005
return-noise overlay. Plus the standard SQL/tool/FS/rules-clause stack so the
agent has to derive strategy from data it can see, not retrieve it from
weight memory.
"""


def render() -> None:
    gr.HTML(_HERO_HTML)
    gr.Plot(value=plots.reward_trajectory(), show_label=False)
    gr.Markdown(_PILLARS_MD)
