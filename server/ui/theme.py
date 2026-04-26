"""Editorial pure-black theme for the TradeBench Gradio UI.

Exports:

- ``DARK_THEME`` — Gradio Base theme with the pure-black palette and
  the Fraunces / DM Sans / IBM Plex Mono font stack.
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
# our component CSS land on the same colors. Pure black canvas.
_NEUTRAL = colors.Color(
    c50="#FFFFFF",
    c100="#F5F5F5",
    c200="#E5E5E5",
    c300="#A3A3A3",
    c400="#737373",
    c500="#525252",
    c600="#404040",
    c700="#262626",
    c800="#171717",
    c900="#0A0A0B",
    c950="#000000",
    name="ink",
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
    neutral_hue=_NEUTRAL,
    font=[
        fonts.GoogleFont("DM Sans"),
        fonts.GoogleFont("IBM Plex Sans"),
        "ui-sans-serif",
        "system-ui",
        "sans-serif",
    ],
    font_mono=[
        fonts.GoogleFont("IBM Plex Mono"),
        fonts.GoogleFont("JetBrains Mono"),
        "SF Mono",
        "Menlo",
        "Consolas",
        "monospace",
    ],
    radius_size=sizes.radius_none,
).set(
    body_background_fill="#000000",
    body_background_fill_dark="#000000",
    body_text_color="#FFFFFF",
    body_text_color_dark="#FFFFFF",
    body_text_color_subdued="rgba(255,255,255,0.66)",
    body_text_color_subdued_dark="rgba(255,255,255,0.66)",
    block_background_fill="#0A0A0B",
    block_background_fill_dark="#0A0A0B",
    block_border_color="rgba(255,255,255,0.10)",
    block_border_color_dark="rgba(255,255,255,0.10)",
    block_label_text_color="#FFFFFF",
    block_label_text_color_dark="#FFFFFF",
    block_title_text_color="#FFFFFF",
    block_title_text_color_dark="#FFFFFF",
    panel_background_fill="#000000",
    panel_background_fill_dark="#000000",
    panel_border_color="rgba(255,255,255,0.10)",
    panel_border_color_dark="rgba(255,255,255,0.10)",
    input_background_fill="#0A0A0B",
    input_background_fill_dark="#0A0A0B",
    input_border_color="rgba(255,255,255,0.10)",
    input_border_color_dark="rgba(255,255,255,0.10)",
    button_primary_background_fill="#FFFFFF",
    button_primary_background_fill_dark="#FFFFFF",
    button_primary_background_fill_hover="#22D3EE",
    button_primary_background_fill_hover_dark="#22D3EE",
    button_primary_text_color="#000000",
    button_primary_text_color_dark="#000000",
    button_secondary_background_fill="transparent",
    button_secondary_background_fill_dark="transparent",
    button_secondary_background_fill_hover="rgba(255,255,255,0.04)",
    button_secondary_background_fill_hover_dark="rgba(255,255,255,0.04)",
    button_secondary_text_color="#FFFFFF",
    button_secondary_text_color_dark="#FFFFFF",
    border_color_primary="rgba(255,255,255,0.10)",
    border_color_primary_dark="rgba(255,255,255,0.10)",
    color_accent_soft="rgba(34,211,238,0.20)",
    color_accent_soft_dark="rgba(34,211,238,0.20)",
)


# Pull serif + body + mono explicitly. RAETH-style stack: Fraunces (display),
# DM Sans (body), IBM Plex Mono (labels). Cormorant Garamond is a softer
# fallback if Fraunces fails.
_FONT_IMPORT = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600&family=Cormorant+Garamond:wght@400;500;600&family=DM+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
"""

NEON_CSS = "\n".join(
    [
        _FONT_IMPORT,
        _read_static("tokens.css"),
        _read_static("components.css"),
    ],
)


__all__ = ["DARK_THEME", "NEON_CSS"]
