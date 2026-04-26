"""Editorial quant-terminal theme for the TradeBench Gradio UI.

Exports:

- ``DARK_THEME`` — Gradio Base theme with the charcoal/cyan palette and
  the Fraunces / IBM Plex Sans / JetBrains Mono font stack.
- ``NEON_CSS`` — inlined design tokens (``static/tokens.css``) and
  component primitives (``static/components.css``).
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr
from gradio.themes.utils import colors, fonts, sizes

_HERE = Path(__file__).resolve().parent


def _read_static(name: str) -> str:
    """Inline a CSS asset from ``server/ui/static/``."""

    path = _HERE / "static" / name
    return path.read_text(encoding="utf-8") if path.is_file() else ""


# Palette values mirror static/tokens.css so Gradio component theming and
# our component CSS land on the same colors.
_CHARCOAL = colors.Color(
    c50="#F8FAFC",
    c100="#E6E8EB",
    c200="#B6BDC7",
    c300="#7A8693",
    c400="#4F5763",
    c500="#3A424E",
    c600="#2A3340",
    c700="#1F2731",
    c800="#1A2029",
    c900="#11161D",
    c950="#0A0E13",
    name="charcoal",
)

_CYAN = colors.Color(
    c50="#ECFEFF",
    c100="#CFFAFE",
    c200="#A5F3FC",
    c300="#67E8F9",
    c400="#22D3EE",
    c500="#06B6D4",
    c600="#0891B2",
    c700="#0E7490",
    c800="#155E75",
    c900="#164E63",
    c950="#083344",
    name="cyan",
)


DARK_THEME = gr.themes.Base(
    primary_hue=_CYAN,
    secondary_hue=_CYAN,
    neutral_hue=_CHARCOAL,
    font=[
        fonts.GoogleFont("IBM Plex Sans"),
        "ui-sans-serif",
        "system-ui",
        "sans-serif",
    ],
    font_mono=[
        fonts.GoogleFont("JetBrains Mono"),
        "SF Mono",
        "Menlo",
        "Consolas",
        "monospace",
    ],
    radius_size=sizes.radius_sm,
).set(
    body_background_fill="#0A0E13",
    body_background_fill_dark="#0A0E13",
    body_text_color="#E6E8EB",
    body_text_color_dark="#E6E8EB",
    body_text_color_subdued="#7A8693",
    body_text_color_subdued_dark="#7A8693",
    block_background_fill="#11161D",
    block_background_fill_dark="#11161D",
    block_border_color="#1F2731",
    block_border_color_dark="#1F2731",
    block_label_text_color="#E6E8EB",
    block_label_text_color_dark="#E6E8EB",
    block_title_text_color="#E6E8EB",
    block_title_text_color_dark="#E6E8EB",
    panel_background_fill="#0A0E13",
    panel_background_fill_dark="#0A0E13",
    panel_border_color="#1F2731",
    panel_border_color_dark="#1F2731",
    input_background_fill="#1A2029",
    input_background_fill_dark="#1A2029",
    input_border_color="#1F2731",
    input_border_color_dark="#1F2731",
    button_primary_background_fill="#22D3EE",
    button_primary_background_fill_dark="#22D3EE",
    button_primary_background_fill_hover="#67E8F9",
    button_primary_background_fill_hover_dark="#67E8F9",
    button_primary_text_color="#0A0E13",
    button_primary_text_color_dark="#0A0E13",
    button_secondary_background_fill="#1A2029",
    button_secondary_background_fill_dark="#1A2029",
    button_secondary_background_fill_hover="#1F2731",
    button_secondary_background_fill_hover_dark="#1F2731",
    button_secondary_text_color="#E6E8EB",
    button_secondary_text_color_dark="#E6E8EB",
    border_color_primary="#1F2731",
    border_color_primary_dark="#1F2731",
    color_accent_soft="#0E7490",
    color_accent_soft_dark="#0E7490",
)


# Fraunces is the one typeface not in Gradio's default font slots; pull it
# explicitly so the editorial headlines render with the variable serif.
_FONT_IMPORT = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600&display=swap');
"""

NEON_CSS = "\n".join(
    [
        _FONT_IMPORT,
        _read_static("tokens.css"),
        _read_static("components.css"),
    ],
)


__all__ = ["DARK_THEME", "NEON_CSS"]
