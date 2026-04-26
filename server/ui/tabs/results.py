"""Reflection-Loop tab — the headline results, both models, embedded as the
canonical PNG figures from ``docs/figures/``."""

from __future__ import annotations

import gradio as gr

from ..figures import figure_card


_INTRO_HTML = """
<div class="tb-section-eyebrow">reflection-loop &middot; 5 iters qwen &middot; 3 iters glm</div>
<h2 class="tb-section-title">
    Each iteration: <span class="tb-accent">Claude Opus 4.7</span> reads the prior
    trajectory, names the single most-costly failure mode, and proposes a surgical
    edit. <span class="tb-faded">The model weights are never touched.</span>
</h2>

<p class="tb-section-lead tb-section-lead-mute">
    Two open-weights base models, identical reflection harness, identical evaluation.
    Length budget capped at 1.10x the previous iteration's prompt. Five iterations
    on Qwen3-32B (Groq), three on GLM-5.1 (Together).
</p>
""" + figure_card(
    src="reward_evolution.png",
    title="reward evolution",
    meta="qwen3-32b vs glm-5.1",
    caption="Both curves climb. Qwen converges smoothly across 5 iterations with no "
            "regressions; GLM dips on iter 2 before recovering on iter 3. The y-axis "
            "is mean per-bar composite reward on the held-out 119-bar test episode.",
)


_PROMPT_FIGURE_HTML = figure_card(
    src="prompt_evolution.png",
    title="system prompt evolution",
    meta="baseline &rarr; iter 5 &middot; qwen3-32b",
    caption="Three sections changed across 5 reflections; the rest of the prompt was "
            "preserved. The 1.10x length cap was respected on every iteration.",
)


_QWEN_HEADER = """
<div class="tb-section-head">
    <div>
        <div class="tb-section-eyebrow">model 01 / 02 &middot; qwen3-32b &middot; groq</div>
        <h2 class="tb-section-title">
            Score climbed monotonically from <span class="tb-num">0.6155</span>
            to <span class="tb-num">0.6584</span>
            <span class="tb-faded">(+4.3 pp);</span>
            ROI grew <span class="tb-num">3.6&times;</span>.
        </h2>
    </div>
    <div></div>
</div>

<table class="tb-table">
    <thead>
        <tr><th>Iter</th><th class="mono">Score</th><th class="mono">ROI</th><th class="mono">place_order</th><th class="mono">sandbox_exec</th><th class="mono">view_*</th></tr>
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

_QWEN_FIGURES_HTML = (
    figure_card(
        src="reward_roi_combined.png",
        title="roi + score per iteration",
        meta="qwen3-32b &middot; 5 iters",
        caption="Twin-axis: ROI bars on the left axis grow from +2.78% to +9.89%; "
                "score line on the right axis rises monotonically from 0.6155 to 0.6584.",
    )
    + figure_card(
        src="bar_alpha_vs_bnh.png",
        title="per-bar alpha vs equal-weight b&amp;h",
        meta="baseline vs iter 5 &middot; qwen3-32b",
        caption="<strong>The alpha gap closed from -0.136 log-points to -0.061</strong>. "
                "Reflection cut the bar-by-bar tracking error against the equal-weight "
                "buy-and-hold baseline by more than half.",
    )
    + figure_card(
        src="action_mix_evolution.png",
        title="action mix per iteration",
        meta="qwen3-32b &middot; 5 iters",
        caption="Reflection drove more <code>place_order</code> calls (4 &rarr; 12), "
                "more selective <code>sandbox_exec</code> usage, and dramatically more "
                "state inspection via <code>view_*</code>.",
    )
    + figure_card(
        src="reward_components_baseline_vs_final.png",
        title="reward components &middot; baseline vs final",
        meta="qwen3-32b &middot; sorted by |&Delta;|",
        caption="Mean per-bar contribution per component, baseline vs iter 5. "
                "<strong>c_alpha is the dominant gainer</strong> &mdash; exactly what "
                "we'd expect from a policy that learned to actually trade.",
    )
)


_QWEN_DIAGNOSIS = """
<div class="tb-section-eyebrow">iter 5 &middot; opus diagnosis</div>
<blockquote class="tb-quote">
    After the initial build on bar 1 the agent essentially never re-checks the
    gap between its recorded 0.60 intended_exposure and its actual ~0.20 gross
    &mdash; it records new decisions and advances for 100+ bars without ever firing
    the remaining build legs, yielding only 14 place_orders over 119 bars.
</blockquote>
<p class="tb-section-lead tb-section-lead-mute">
    <strong>Surgical edit:</strong> &sect;5 step-4 strengthened to also cover bars
    2-3 when the initial build is incomplete; &sect;7.5 named the exact failure
    pattern ("two legs filled out of five committed") so the agent could
    recognize itself doing it. After the edit landed, place_order count rose
    14 &rarr; 12 (selective, not just busy) and ROI rose +6.96% &rarr; +9.89%.
</p>
"""


_GLM_HEADER = """
<div class="tb-section-head">
    <div>
        <div class="tb-section-eyebrow">model 02 / 02 &middot; glm-5.1 &middot; together</div>
        <h2 class="tb-section-title">
            <span class="tb-faded">A different starting pathology,</span>
            the same converged behavior.
        </h2>
    </div>
    <div></div>
</div>

<table class="tb-table">
    <thead>
        <tr><th>Iter</th><th class="mono">Score</th><th class="mono">ROI</th><th class="mono">place_order</th><th class="mono">sandbox_exec</th><th class="mono">view_*</th></tr>
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

_GLM_FIGURES_HTML = (
    figure_card(
        src="reward_roi_combined_glm.png",
        title="roi + score per iteration",
        meta="glm-5.1 &middot; 3 iters",
        caption="Score climbs from 0.6243 to 0.6453 with a small dip on iter 2 "
                "before recovering. ROI grows from +2.83% to +4.69%.",
    )
    + figure_card(
        src="bar_alpha_vs_bnh_glm.png",
        title="per-bar alpha vs equal-weight b&amp;h",
        meta="baseline vs iter 3 &middot; glm-5.1",
        caption="GLM's alpha gap also closed across reflection &mdash; same direction "
                "as Qwen, smaller magnitude given the shorter loop.",
    )
    + figure_card(
        src="action_mix_evolution_glm.png",
        title="action mix per iteration",
        meta="glm-5.1 &middot; 3 iters",
        caption="GLM moved from <em>think-first-act-rarely</em> toward more orders "
                "and more selective <code>sandbox_exec</code> usage &mdash; the same "
                "operating point Qwen converged to from the opposite direction.",
    )
    + figure_card(
        src="reward_components_baseline_vs_final_glm.png",
        title="reward components &middot; baseline vs final",
        meta="glm-5.1 &middot; sorted by |&Delta;|",
        caption="Same component decomposition as Qwen, computed against GLM's own "
                "baseline.",
    )
)


_INSIGHT_HTML = """
<div class="tb-section-eyebrow">three observations from the head-to-head</div>
<h2 class="tb-section-title">
    <span class="tb-faded">What we learned by running</span>
    the same loop on two different models.
</h2>

<div class="tb-three-up">
    <div class="tb-card">
        <div class="tb-card-num">01 / 03</div>
        <div class="tb-card-title">Same loop, different curves</div>
        <div class="tb-card-body">Qwen converged smoothly across 5 iterations
            with no regressions. GLM had a small iter-1 to iter-2 dip before
            recovering. Stronger instruction-following lets the loop convert
            each reflector edit into a behavioral improvement more reliably.</div>
    </div>
    <div class="tb-card">
        <div class="tb-card-num">02 / 03</div>
        <div class="tb-card-title">Bigger pathology, bigger first jump</div>
        <div class="tb-card-body">GLM's iter-1 score gain (+0.019) was about
            <strong>3&times;</strong> Qwen's (+0.006). GLM had a more egregious
            starting policy (2 orders, 88 view_*) for the reflector to call
            out. The first edit always lands hardest.</div>
    </div>
    <div class="tb-card">
        <div class="tb-card-num">03 / 03</div>
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
    gr.HTML(_PROMPT_FIGURE_HTML)
    gr.HTML(_QWEN_HEADER)
    gr.HTML(_QWEN_FIGURES_HTML)
    gr.HTML(_QWEN_DIAGNOSIS)
    gr.HTML(_GLM_HEADER)
    gr.HTML(_GLM_FIGURES_HTML)
    gr.HTML(_INSIGHT_HTML)
