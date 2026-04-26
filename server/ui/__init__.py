"""Gradio web UI for TradeBench.

Public surface intentionally minimal: ``build_ui`` constructs the
``gr.Blocks`` app, ``DARK_THEME`` and ``NEON_CSS`` are exposed for the
``gr.mount_gradio_app`` call in ``server/app.py``.
"""

from .build_ui import build_ui
from .theme import DARK_THEME, NEON_CSS

__all__ = ["build_ui", "DARK_THEME", "NEON_CSS"]
