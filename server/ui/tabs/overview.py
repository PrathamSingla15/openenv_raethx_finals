"""Introduction tab — minimal landing page.

Just the hook headline and the "Why this matters" content from Blog.md § 0.
Tight spacing — designed to fit one viewport without scrolling.
"""

from __future__ import annotations

import gradio as gr


_LANDING_HTML = """
<section class="tb-intro">
    <div class="tb-intro-eyebrow">
        <span class="is-accent">OpenEnv Hackathon</span>
        <span>&middot;</span>
        <span>Theme 2 &middot; Long-horizon planning</span>
    </div>

    <h1 class="tb-intro-title">
        If AI models are <span class="tb-accent">so smart</span>, why aren't they rich?
    </h1>

    <div class="tb-intro-rule"></div>

    <div class="tb-intro-tag">&mdash; Why it matters</div>

    <div class="tb-intro-body">
        <p>
            Capital allocation drives the economy. It decides which drugs get
            developed, which technologies get built, and which ideas survive
            long enough to matter.
        </p>
        <p>
            And yet, for something so central, it still runs on a fragile
            foundation: human judgment, plus a thin and badly instrumented
            layer of agentic systems that pretend to know what they are doing.
            Frontier LLMs can reason about a single trade in isolation, but
            break down across hundreds of sequential decisions where past
            actions reshape the future state distribution. The skill that
            fails is not analytical depth. It is execution discipline:
            <em>when to act, when to abstain, how to size against existing
            positions, how to keep from drifting into a regime the agent did
            not intend.</em>
        </p>
        <p>
            What was once left to individual judgement can become an
            optimizable system. A new hill to climb.
        </p>
        <p>
            That is the bet TradeBench makes. We turn long-horizon equities
            trading into an environment where execution discipline is a
            measurable, decomposable, gradient-friendly target, not a soft
            skill anyone has to grade by feel. The reward is bounded in
            <code>[0, 1]</code>, decomposed across seven trader-recognizable
            components, and gated by a binary compliance check. The data is
            real OHLCV behind a four-layer anti-memorization stack so the
            agent has to derive strategy from what it sees, not retrieve it
            from training memory.
        </p>
    </div>
</section>

<style>
.tb-intro {
    max-width: 960px;
    margin: 0 auto;
    padding: 32px 0 24px;
}
.tb-intro-eyebrow {
    font-family: var(--type-mono);
    font-size: 11px;
    color: var(--ink-faint);
    letter-spacing: 0.18em;
    text-transform: uppercase;
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    align-items: center;
    margin: 0 0 18px;
}
.tb-intro-eyebrow .is-accent { color: var(--accent); }
.tb-intro-title {
    font-family: var(--type-body);
    font-weight: 700;
    font-size: clamp(2.8rem, 7.2vw, 5.6rem);
    line-height: 1.02;
    letter-spacing: -0.035em;
    color: var(--ink);
    margin: 0 0 28px;
    max-width: 18ch;
}
.tb-intro-title .tb-accent {
    color: var(--ink);
    text-decoration: underline;
    text-decoration-color: var(--accent);
    text-decoration-thickness: 5px;
    text-underline-offset: 10px;
}
.tb-intro-rule {
    height: 1px;
    background: var(--rule);
    margin: 0 0 20px;
}
.tb-intro-tag {
    font-family: var(--type-mono);
    font-size: 11px;
    color: var(--ink-faint);
    letter-spacing: 0.22em;
    text-transform: uppercase;
    margin: 0 0 12px;
}
.tb-intro-body p {
    font-family: var(--type-body);
    font-size: 0.96rem;
    line-height: 1.55;
    color: var(--ink-mute);
    margin: 0 0 12px;
    max-width: 70ch;
}
.tb-intro-body p:last-child { margin-bottom: 0; }
.tb-intro-body em {
    font-style: normal;
    font-weight: 600;
    color: var(--ink);
}
.tb-intro-body code {
    color: var(--accent-hi);
    background: rgba(34, 211, 238, 0.06);
    padding: 1px 5px;
    border: 1px solid var(--accent-lo);
    font-size: 0.92em;
}
@media (max-width: 720px) {
    .tb-intro { padding: 24px 0 16px; }
    .tb-intro-title { font-size: clamp(2rem, 10vw, 3rem); letter-spacing: -0.03em; }
    .tb-intro-title .tb-accent { text-decoration-thickness: 3px; text-underline-offset: 7px; }
    .tb-intro-body p { font-size: 0.92rem; line-height: 1.5; }
}
</style>
"""


def render() -> None:
    """Render the Introduction landing page."""
    gr.HTML(_LANDING_HTML)
