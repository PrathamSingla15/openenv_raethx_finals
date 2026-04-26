"""Top-level Gradio Blocks app for TradeBench.

Editorial layout: sticky frosted nav at top, full-viewport landing scroll on the
Trajectory tab (raeth.ai-style), then deep-dive tabs preserved beneath.

The landing's reveal/parallax script lives at ``/ui-static/motion.js`` (mounted
by ``server.app._mount_ui_static``). All CSS is inlined via ``theme.NEON_CSS``.
"""

from __future__ import annotations

import gradio as gr

from .tabs import (
    baselines,
    demo,
    docs,
    environment,
    overview,
    results,
    rewards,
    walk_forward,
)
from .theme import DARK_THEME, NEON_CSS

# Sticky frosted nav. Lockup (left) + numbered anchors (center, ≥980px) +
# cyan ghost CTA (right). Anchors point to numbered sections inside the
# Trajectory tab; motion.js handles smooth-scroll. The "Get the code" CTA
# is a real external link to the GitHub repo.
_NAV_HTML = """
<div class="tb-nav">
    <div class="tb-nav-row">
        <a class="tb-nav-lockup" href="#tb-section-hero" data-tb-tab="Trajectory">
            TradeBench
            <span class="tb-nav-lockup-mute">/ OpenEnv Theme 2</span>
        </a>
        <nav class="tb-nav-anchors" aria-label="Sections">
            <a class="tb-nav-anchor" href="#tb-section-01" data-tb-tab="Trajectory">
                <span class="tb-nav-anchor-num">01</span>Problem
            </a>
            <a class="tb-nav-anchor" href="#tb-section-02" data-tb-tab="Trajectory">
                <span class="tb-nav-anchor-num">02</span>Environment
            </a>
            <a class="tb-nav-anchor" href="#tb-section-03" data-tb-tab="Trajectory">
                <span class="tb-nav-anchor-num">03</span>Reward
            </a>
            <a class="tb-nav-anchor" href="#tb-section-04" data-tb-tab="Trajectory">
                <span class="tb-nav-anchor-num">04</span>Defenses
            </a>
            <a class="tb-nav-anchor" href="#tb-section-05" data-tb-tab="Trajectory">
                <span class="tb-nav-anchor-num">05</span>Results
            </a>
            <a class="tb-nav-anchor" href="#tb-section-06" data-tb-tab="Trajectory">
                <span class="tb-nav-anchor-num">06</span>Run&nbsp;it
            </a>
        </nav>
        <a class="tb-nav-cta"
           href="https://github.com/PrathamSingla15/openenv_raethx_finals"
           target="_blank" rel="noreferrer">
            Get&nbsp;the&nbsp;code&nbsp;<span aria-hidden="true">&rarr;</span>
        </a>
    </div>
</div>
"""

# Landing motion.js loads from the FastAPI /ui-static mount. Defer keeps the
# script non-blocking; the script is self-init via DOMContentLoaded.
_MOTION_SCRIPT = """
<script src="/ui-static/motion.js" defer></script>
"""


def build_ui() -> gr.Blocks:
    """Construct and return the multi-tab TradeBench Gradio app."""

    # Gradio 5 takes theme/css at Blocks construction; Gradio 6 reads them
    # at mount time. Probe the signature so we launch cleanly on either.
    import inspect

    blocks_kwargs: dict = {"title": "TradeBench"}
    if "theme" in inspect.signature(gr.Blocks.__init__).parameters:
        blocks_kwargs["theme"] = DARK_THEME
    if "css" in inspect.signature(gr.Blocks.__init__).parameters:
        blocks_kwargs["css"] = NEON_CSS

    with gr.Blocks(**blocks_kwargs) as app:
        gr.HTML(_NAV_HTML)
        gr.HTML(_MOTION_SCRIPT)

        with gr.Tabs():
            with gr.Tab("Trajectory", id="trajectory"):
                overview.render()
            with gr.Tab("Reflection Loop", id="reflection_loop"):
                results.render()
            with gr.Tab("Reward", id="reward"):
                rewards.render()
            with gr.Tab("Environment", id="environment"):
                environment.render()
            with gr.Tab("Defenses", id="defenses"):
                walk_forward.render()
            with gr.Tab("Baselines", id="baselines"):
                baselines.render()
            with gr.Tab("Run", id="demo"):
                demo.render()
            with gr.Tab("Docs", id="docs"):
                docs.render()

    return app
