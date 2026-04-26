"""Top-level Gradio Blocks app for TradeBench.

Tab spine: trajectory · environment · reward · defenses · reflection-loop ·
demo · baselines · docs.
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

# Editorial header: paired with .tb-header CSS in components.css.
_HEADER_HTML = """
<div class="tb-header">
    <div class="tb-header-rule"></div>
    <div class="tb-header-meta">
        <span class="tb-header-tag">openenv</span>
        <span class="tb-header-dot">·</span>
        <span class="tb-header-tag">theme 2 · long-horizon planning</span>
        <span class="tb-header-dot">·</span>
        <span class="tb-header-tag">7-component composite reward</span>
        <span class="tb-header-dot">·</span>
        <span class="tb-header-tag">6 anti-leak layers</span>
    </div>
    <h1 class="tb-header-title">Trade<span class="tb-accent">Bench</span></h1>
    <p class="tb-lead">
        A long-horizon, non-stationary, reward-hack-resistant equities-trading RL
        environment for LLM agents. Reward in <code>[0, 1]</code>. Real OHLCV
        behind a four-layer anti-memorization stack. Designed for actual
        training, not just evaluation.
    </p>
</div>
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
        gr.HTML(_HEADER_HTML)

        with gr.Tabs():
            with gr.Tab("Trajectory", id="trajectory"):
                overview.render()
            with gr.Tab("Environment", id="environment"):
                environment.render()
            with gr.Tab("Reward", id="reward"):
                rewards.render()
            with gr.Tab("Defenses", id="defenses"):
                walk_forward.render()
            with gr.Tab("Reflection Loop", id="reflection_loop"):
                results.render()
            with gr.Tab("Run", id="demo"):
                demo.render()
            with gr.Tab("Baselines", id="baselines"):
                baselines.render()
            with gr.Tab("Docs", id="docs"):
                docs.render()

    return app
