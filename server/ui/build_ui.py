"""Top-level Gradio Blocks app for TradeBench.

Tab spine: overview · environment · rewards · walk_forward · demo ·
baselines · docs.
"""

from __future__ import annotations

import gradio as gr

from .tabs import (
    baselines,
    demo,
    docs,
    environment,
    overview,
    rewards,
    walk_forward,
)
from .theme import DARK_THEME, NEON_CSS

# Editorial header — paired with .tb-header CSS in components.css.
_HEADER_HTML = """
<div class="tb-header">
    <h1>Trade<span class="tb-accent">Bench</span></h1>
    <p class="tb-lead">
        A long-horizon, non-stationary, reward-hack-resistant finance
        trading RL environment for LLM agents — built for actual training,
        not just evaluation.
    </p>
    <div class="tb-meta">
        openenv · theme 2 · 7-component composite reward · layered look-ahead defense
    </div>
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
            with gr.Tab("Overview", id="overview"):
                overview.render()
            with gr.Tab("Environment", id="environment"):
                environment.render()
            with gr.Tab("Rewards", id="rewards"):
                rewards.render()
            with gr.Tab("Walk-Forward", id="walk_forward"):
                walk_forward.render()
            with gr.Tab("Demo", id="demo"):
                demo.render()
            with gr.Tab("Baselines", id="baselines"):
                baselines.render()
            with gr.Tab("Docs & Links", id="docs"):
                docs.render()

    return app
