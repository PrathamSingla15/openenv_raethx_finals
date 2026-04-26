"""Tab modules for the TradeBench Gradio UI.

Each module exposes a single ``render()`` function that drops its content
into the active Gradio context.
"""

from . import (
    baselines,
    demo,
    docs,
    environment,
    overview,
    results,
    rewards,
    walk_forward,
)

__all__ = [
    "baselines",
    "demo",
    "docs",
    "environment",
    "overview",
    "results",
    "rewards",
    "walk_forward",
]
