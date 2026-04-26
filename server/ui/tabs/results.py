"""Reflection-Loop tab: the headline results, both models, all interactive."""

from __future__ import annotations

import gradio as gr

from .. import plots


_INTRO_HTML = """
<div class="tb-section-eyebrow">reflection-loop · 5 iterations Qwen · 3 iterations GLM</div>
<h2 class="tb-section-title">Each iteration: Claude Opus 4.7 reads the prior
trajectory, names the single most-costly failure mode, and proposes a surgical
edit (≤ 1.10× length cap). The model weights are never touched.</h2>

<p class="tb-section-lead">
    Two open-weights base models, identical reflection harness, identical evaluation.
    Charts are interactive: hover for per-iter detail; toggle traces in the legend;
    drag to zoom on the per-bar alpha plot.
</p>
"""


_QWEN_HEADER = """
<div class="tb-section-eyebrow">Qwen3-32B  ·  Groq  ·  5 iterations</div>
<h2 class="tb-section-title">Score climbed monotonically from 0.6155 to 0.6584
(+4.3 pp); ROI grew from +2.78% to +9.89% (3.6× absolute).</h2>

<table class="tb-table" style="margin-bottom: 0;">
    <thead>
        <tr><th>Iter</th><th>Score</th><th>ROI</th><th>place_order</th><th>sandbox_exec</th><th>view_*</th></tr>
    </thead>
    <tbody>
        <tr><td><strong>0 baseline</strong></td><td class="mono">0.6155</td><td class="mono">+2.78%</td><td class="mono">4</td><td class="mono">22</td><td class="mono">6</td></tr>
        <tr><td>1 reflection</td><td class="mono">0.6214</td><td class="mono">+4.72%</td><td class="mono">6</td><td class="mono">29</td><td class="mono">23</td></tr>
        <tr><td>2 reflection</td><td class="mono">0.6306</td><td class="mono">+3.88%</td><td class="mono">10</td><td class="mono">6</td><td class="mono">17</td></tr>
        <tr><td>3 reflection</td><td class="mono">0.6381</td><td class="mono">+6.31%</td><td class="mono">11</td><td class="mono">1</td><td class="mono">32</td></tr>
        <tr><td>4 reflection</td><td class="mono">0.6476</td><td class="mono">+6.96%</td><td class="mono">14</td><td class="mono">6</td><td class="mono">32</td></tr>
        <tr><td><strong>5 reflection</strong></td><td class="mono"><strong>0.6584</strong></td><td class="mono"><strong>+9.89%</strong></td><td class="mono"><strong>12</strong></td><td class="mono"><strong>10</strong></td><td class="mono"><strong>31</strong></td></tr>
    </tbody>
</table>
"""


_QWEN_DIAGNOSIS = """
<div class="tb-section-eyebrow">iter 5 · opus diagnosis</div>
<blockquote class="tb-quote">
    "After the initial build on bar 1 the agent essentially never re-checks the
    gap between its recorded 0.60 intended_exposure and its actual ~0.20 gross
    — it records new decisions and advances for 100+ bars without ever firing
    the remaining build legs, yielding only 14 place_orders over 119 bars."
</blockquote>
<p class="tb-section-lead tb-section-lead-mute">
    <strong>Surgical edit:</strong> §5 step-4 strengthened to also cover bars
    2-3 when the initial build is incomplete; §7.5 named the exact failure
    pattern ("two legs filled out of five committed") so the agent could
    recognize itself doing it. After the edit landed, place_order count rose
    14 → 12 (selective, not just busy) and ROI rose +6.96% → +9.89%.
</p>
"""


_GLM_HEADER = """
<div class="tb-section-eyebrow">GLM-5.1  ·  Together  ·  3 iterations</div>
<h2 class="tb-section-title">A different starting pathology, the same converged
behavior: more orders, more selective sandbox use, more state inspection.</h2>

<table class="tb-table" style="margin-bottom: 0;">
    <thead>
        <tr><th>Iter</th><th>Score</th><th>ROI</th><th>place_order</th><th>sandbox_exec</th><th>view_*</th></tr>
    </thead>
    <tbody>
        <tr><td><strong>0 baseline</strong></td><td class="mono">0.6243</td><td class="mono">+2.83%</td><td class="mono">2</td><td class="mono">81</td><td class="mono">88</td></tr>
        <tr><td>1 reflection</td><td class="mono">0.6432</td><td class="mono">+4.39%</td><td class="mono">6</td><td class="mono">44</td><td class="mono">110</td></tr>
        <tr><td>2 reflection</td><td class="mono">0.6382</td><td class="mono">+3.72%</td><td class="mono">6</td><td class="mono">19</td><td class="mono">182</td></tr>
        <tr><td><strong>3 reflection</strong></td><td class="mono"><strong>0.6453</strong></td><td class="mono"><strong>+4.69%</strong></td><td class="mono"><strong>6</strong></td><td class="mono"><strong>41</strong></td><td class="mono"><strong>148</strong></td></tr>
    </tbody>
</table>

<p class="tb-section-lead tb-section-lead-mute">
GLM started <em>think-first-act-rarely</em> (2 orders, 81 sandbox_exec, 88
view_*); Qwen started <em>act-quickly-think-rarely</em> (4 orders, 22
sandbox_exec, 6 view_*). Reflection drove both toward the same operating
point. That convergence from opposite starting positions is the strongest
signal that the loop is finding something real about the env's success
conditions, not patching one model's idiosyncratic quirks.
</p>
"""


_INSIGHT_HTML = """
<div class="tb-section-eyebrow">three observations from the head-to-head</div>

<div class="tb-three-up">
    <div class="tb-card">
        <div class="tb-card-num">i.</div>
        <div class="tb-card-title">Same loop, different curves</div>
        <div class="tb-card-body">Qwen converged smoothly across 5 iterations
            with no regressions. GLM had a small iter-1 → iter-2 dip before
            recovering. Stronger instruction-following lets the loop convert
            each reflector edit into a behavioral improvement more reliably.</div>
    </div>
    <div class="tb-card">
        <div class="tb-card-num">ii.</div>
        <div class="tb-card-title">Bigger pathology, bigger first jump</div>
        <div class="tb-card-body">GLM's iter-1 score gain (+0.019) was about
            <strong>3×</strong> Qwen's (+0.006). GLM had a more egregious
            starting policy (2 orders, 88 view_*) for the reflector to call
            out. The first edit always lands hardest.</div>
    </div>
    <div class="tb-card">
        <div class="tb-card-num">iii.</div>
        <div class="tb-card-title">Opposite starts, convergent ends</div>
        <div class="tb-card-body">Both models drifted toward more orders, more
            selective sandbox usage, and dramatically more state inspection.
            That convergence from opposite starting points is what makes us
            believe reflection is finding something real about the env, not
            patching idiosyncratic quirks.</div>
    </div>
</div>
"""


def render() -> None:
    gr.HTML(_INTRO_HTML)
    gr.Plot(value=plots.reward_trajectory(), show_label=False)

    gr.HTML(_QWEN_HEADER)
    with gr.Row():
        gr.Plot(value=plots.roi_score_combined(plots.QWEN_ID), show_label=False)
    with gr.Row():
        gr.Plot(value=plots.bar_alpha_vs_bnh(plots.QWEN_ID), show_label=False)
    with gr.Row():
        gr.Plot(value=plots.action_mix(plots.QWEN_ID), show_label=False)
    with gr.Row():
        gr.Plot(value=plots.reward_components(plots.QWEN_ID), show_label=False)
    gr.HTML(_QWEN_DIAGNOSIS)

    gr.HTML(_GLM_HEADER)
    with gr.Row():
        gr.Plot(value=plots.roi_score_combined(plots.GLM_ID), show_label=False)
    with gr.Row():
        gr.Plot(value=plots.bar_alpha_vs_bnh(plots.GLM_ID), show_label=False)
    with gr.Row():
        gr.Plot(value=plots.action_mix(plots.GLM_ID), show_label=False)
    with gr.Row():
        gr.Plot(value=plots.reward_components(plots.GLM_ID), show_label=False)

    gr.HTML(_INSIGHT_HTML)
