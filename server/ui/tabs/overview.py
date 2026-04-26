"""Trajectory tab — minimal landing page.

Just the hook headline and the "Why this matters" content from Blog.md § 0.
No numbered sections, no plots, no metrics strip, no CTAs. Deep audit lives
in the other tabs.
"""

from __future__ import annotations

import gradio as gr


_LANDING_HTML = """
<section id="tb-section-hero" class="tb-landing">
    <div class="tb-landing-eyebrow">
        <span class="is-accent">OpenEnv Hackathon</span>
        <span class="tb-dot">/</span>
        <span>Theme 2 &middot; Long-horizon planning</span>
    </div>

    <h1 class="tb-landing-title">
        If AI models are <span class="tb-accent">so smart</span>, why aren't they rich?
    </h1>
</section>

<section id="tb-section-why" class="tb-numbered-section">
    <div class="tb-numbered-head">
        <div class="tb-numbered-tag">
            <span class="is-accent">&mdash;</span> Why it matters
        </div>
        <div>
            <p class="tb-numbered-sub">
                Capital allocation drives the economy. It decides which drugs get
                developed, which technologies get built, and which ideas survive
                long enough to matter.
            </p>
            <p class="tb-numbered-sub">
                And yet, for something so central, it still runs on a fragile
                foundation: human judgment, plus a thin and badly instrumented
                layer of agentic systems that pretend to know what they are
                doing. Frontier LLMs can reason about a single trade in
                isolation, but break down across hundreds of sequential decisions
                where past actions reshape the future state distribution. The
                skill that fails is not analytical depth. It is execution
                discipline: <em>when to act, when to abstain, how to size
                against existing positions, how to keep from drifting into a
                regime the agent did not intend.</em>
            </p>
            <p class="tb-numbered-sub">
                What was once left to individual judgement can become an
                optimizable system. A new hill to climb.
            </p>
            <p class="tb-numbered-sub">
                That is the bet TradeBench makes. We turn long-horizon equities
                trading into an environment where execution discipline is a
                measurable, decomposable, gradient-friendly target, not a soft
                skill anyone has to grade by feel. The reward is bounded in
                <code>[0, 1]</code>, decomposed across seven trader-recognizable
                components, and gated by a binary compliance check. The data is
                real OHLCV behind a four-layer anti-memorization stack so the
                agent has to derive strategy from what it sees, not retrieve it
                from training memory. The episode is 119 bars long, ~50&ndash;200
                tool calls deep, and emits a dense per-bar reward designed for
                any optimizer that can use a scalar signal: prompt evolution,
                GRPO, PPO, or anything that comes next.
            </p>
        </div>
    </div>
</section>
"""


def render() -> None:
    """Render the minimal Trajectory landing page."""
    gr.HTML(_LANDING_HTML)
