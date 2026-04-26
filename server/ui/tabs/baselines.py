"""Baselines tab: three deterministic reference strategies for the
trained-vs-untrained comparison."""

from __future__ import annotations

import gradio as gr


_INTRO_HTML = """
<div class="tb-section-eyebrow">three reference strategies</div>
<h2 class="tb-section-title">
    <span class="tb-faded">Baselines establish the floor</span>
    every reflection-loop result is implicitly measured against.
</h2>

<p class="tb-section-lead">
All three are deterministic, async coroutines under
<code>tradebench.baselines</code>. Drive them via
<code>uv run python inference.py --baseline {cash,equal_weight,kelly}</code>;
each emits the same trajectory shape as the LLM driver, so the verifier
parses either output identically.
</p>

<table class="tb-table">
    <thead>
        <tr><th>Baseline</th><th>Exposure</th><th>Allocation</th><th>Rebalance</th></tr>
    </thead>
    <tbody>
        <tr><td><code>cash</code></td><td class="mono">0%</td><td>&mdash;</td><td>&mdash;</td></tr>
        <tr><td><code>equal_weight</code></td><td class="mono">90%</td><td>1/N across the visible universe</td><td>One-shot at episode start</td></tr>
        <tr><td><code>kelly</code></td><td class="mono">50%</td><td>1/N &times; half-Kelly fraction</td><td>One-shot at episode start</td></tr>
    </tbody>
</table>
"""


_REFERENCE_HTML = """
<div class="tb-section-eyebrow">benchmark reference</div>
<h2 class="tb-section-title">
    Equal-weight buy-and-hold <span class="tb-faded">sets the bar.</span>
</h2>

<p class="tb-section-lead">
On the same 119-bar test episode the reflection-trained Qwen3-32B reaches
<code>+9.89%</code> ROI; equal-weight buy-and-hold over the same 10 assets
returns <code>+15.96%</code>. The trained agent still trails B&amp;H, but the
gap closed from roughly <code>&minus;13 log-points</code> at baseline to
<code>&minus;5 log-points</code> at iter 5 &mdash; the single most informative
chart on the Reflection-Loop tab.
</p>

<p class="tb-section-lead tb-section-lead-mute">
The honest read: prompt-only optimization can install execution discipline
that the base model already has the knowledge to use, but it cannot
manufacture alpha that exceeds what a static do-nothing portfolio captures
in a positive-drift regime. To beat B&amp;H you likely need a stronger base
model or actual gradient-based RL. Reflection gets you a much better
starting point; it does not replace gradient signal forever.
</p>
"""


_NEXT_HTML = """
<div class="tb-section-eyebrow">natural follow-ons</div>
<h2 class="tb-section-title">
    Reflection as <span class="tb-faded">warm-start,</span>
    RL as long-horizon refinement.
</h2>

<div class="tb-three-up">
    <div class="tb-card">
        <div class="tb-card-num">01 / 03</div>
        <div class="tb-card-title">Reflection-as-init</div>
        <div class="tb-card-body">Use the iter-5 prompt as the starting policy
            for an actual GRPO/PPO run. The training loop reads a calibrated
            executor instead of a cold one, which collapses the exploration
            phase and recovers the gradient signal earlier.</div>
    </div>
    <div class="tb-card">
        <div class="tb-card-num">02 / 03</div>
        <div class="tb-card-title">Multi-window cross-validation</div>
        <div class="tb-card-body">Currently every result is from one 119-bar
            window. Re-running across multiple randomly drawn source windows
            lets the claim be "generalization" rather than "fit on one regime".</div>
    </div>
    <div class="tb-card">
        <div class="tb-card-num">03 / 03</div>
        <div class="tb-card-title">Stronger reflector, weaker base</div>
        <div class="tb-card-body">Sweep the (reflector, agent) combinations to
            characterize how reflection scales with reflector capability and
            base-model instruction-following. Today's run uses Opus 4.7 against
            Qwen3-32B and GLM-5.1.</div>
    </div>
</div>
"""


def render() -> None:
    gr.HTML(_INTRO_HTML)
    gr.HTML(_REFERENCE_HTML)
    gr.HTML(_NEXT_HTML)
