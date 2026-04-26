"""Viz C · system prompt evolution as a draw.io diagram.

Generates a presentation-grade conceptual diagram showing the three
sections of the system prompt that changed during the 5-iteration
reflection loop. Writes mxGraph XML to:

    docs/figures/prompt_evolution.drawio.xml

The XML can be opened in the draw.io editor (via mcp__drawio__open_drawio_xml
or by uploading to https://app.diagrams.net) and exported to PNG. The
exported PNG goes to docs/figures/prompt_evolution.png.

Why a diagram and not a literal text diff: the baseline prompt is 399
lines and the iter-5 prompt is 445 lines, with most lines unchanged.
A literal monospace dump makes the actual changes invisible. This
diagram surfaces only the three sections that materially changed.
"""

from __future__ import annotations

from pathlib import Path

from scripts.visualize.data_loader import PROJECT_ROOT


# mxGraph XML: a single page with title, summary cards, three before/after
# pairs, and an "untouched" footer. Coordinates are absolute pixels in the
# diagram's coordinate space; draw.io renders SVG/PNG from these directly.

_BG_TITLE = "#f5f5f5"
_RED_FILL = "#ffebee"
_RED_STROKE = "#c62828"
_GREEN_FILL = "#e8f5e9"
_GREEN_STROKE = "#2e7d32"
_GRAY = "#757575"
_DARK = "#212121"

# Layout grid (single coordinate system, no nested cells).
W = 1280
COL_LEFT_X = 60
COL_LEFT_W = 540
COL_RIGHT_X = 700
COL_RIGHT_W = 540
ARROW_X1 = COL_LEFT_X + COL_LEFT_W
ARROW_X2 = COL_RIGHT_X


def _esc(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _cell(*, cid: int, value: str, x: int, y: int, w: int, h: int,
          fill: str, stroke: str, font_size: int = 12,
          bold: bool = False, mono: bool = False, align: str = "left",
          v_align: str = "middle") -> str:
    style_parts = [
        "rounded=1",
        "whiteSpace=wrap",
        "html=1",
        f"fillColor={fill}",
        f"strokeColor={stroke}",
        f"strokeWidth=1.5",
        "spacing=12",
        f"align={align}",
        f"verticalAlign={v_align}",
        f"fontSize={font_size}",
        f"fontColor={_DARK}",
    ]
    if bold:
        style_parts.append("fontStyle=1")
    if mono:
        style_parts.append("fontFamily=Courier New")
    style = ";".join(style_parts) + ";"
    return (
        f'<mxCell id="{cid}" value="{_esc(value)}" style="{style}" '
        f'vertex="1" parent="1">'
        f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry" />'
        f"</mxCell>"
    )


def _arrow(*, cid: int, source_id: int, target_id: int) -> str:
    style = (
        "endArrow=classic;html=1;rounded=0;strokeWidth=2;"
        f"strokeColor={_GRAY};fontSize=11;"
    )
    return (
        f'<mxCell id="{cid}" style="{style}" edge="1" parent="1" '
        f'source="{source_id}" target="{target_id}">'
        f'<mxGeometry relative="1" as="geometry" />'
        f"</mxCell>"
    )


_CONFIG_QWEN = {
    "title": "System prompt evolution  ·  Qwen3-32B reflection loop (baseline → iter 5)",
    "subtitle": (
        "5 reflection iterations under a 1.10x length cap. "
        "Three sections changed; the rest of the prompt was preserved."
    ),
    "baseline_card": "Baseline (iter 0)\n399 lines",
    "final_card": "After iter 5\n445 lines  (1.10x cap respected)",
    "section_header": "Three sections changed (everything else untouched):",
    "changes": [
        {
            "label": "§5 step 4  ·  when to place orders",
            "before": (
                "place_order ZERO OR MORE times. Submit one place_order per asset "
                "you want to buy or sell this bar.\n\n"
                "No coupling between recorded convictions and required orders."
            ),
            "after": (
                "For each asset in top_convictions, compute current_weight and "
                "compare to target.\n\n"
                "TRIGGER a rebalance order this bar if any leg has current_weight "
                "< 0.5 * target, OR if drift gap > 2%. A ~0-share conviction is "
                "the largest possible gap."
            ),
        },
        {
            "label": "§7.5  ·  common mistakes the agent must self-recognize",
            "before": (
                "Standing still. Five consecutive non-advance calls means you are "
                "not making progress. Recover by calling advance_day.\n\n"
                "Only one symptom named."
            ),
            "after": (
                "Standing still has TWO failure modes:\n"
                "  1. Five consecutive non-advance calls.\n"
                "  2. SYMMETRIC: advancing bar after bar while target basket "
                "remains unfilled. Two legs filled out of five committed is "
                "standing still with the clock moving.\n\n"
                "Recording without sizing the implied orders is also a form "
                "of standing still."
            ),
        },
        {
            "label": "§6 worked example  ·  the iteration discipline",
            "before": (
                "Response 7 (settle and advance):\n"
                "  advance_day\n\n"
                "Worked example shows initial-build pattern, no caveat about "
                "checking residual gaps."
            ),
            "after": (
                "Response 7 (settle and advance · only after verifying no "
                "conviction leg still has current_weight < 0.5 * target_weight):\n"
                "  advance_day\n\n"
                "Caution: on later bars, run the gap-check before advancing."
            ),
        },
    ],
    "footer": (
        "Untouched across all 5 reflections:  anti-memorization rules-clause  ·  "
        "tool surface  ·  tier definitions  ·  output contract  ·  sandbox spec.\n"
        "Opus correctly identified these as already working and did not spend "
        "the 1.10x length budget rewriting them."
    ),
}

_CONFIG_GLM = {
    "title": "System prompt evolution  ·  GLM-5.1 reflection loop (baseline → iter 3)",
    "subtitle": (
        "3 reflection iterations under a 1.10x length cap. "
        "Three sections changed; the rest of the prompt was preserved."
    ),
    "baseline_card": "Baseline (iter 0)\n399 lines",
    "final_card": "After iter 3\n443 lines  (1.10x cap respected)",
    "section_header": "Three sections changed (everything else untouched):",
    "changes": [
        {
            "label": "§4.2 record_decision  ·  intent vs action",
            "before": (
                "CRITICAL: record_decision ONLY logs your intent for audit. "
                "It does NOT place any orders, does NOT move money. To actually "
                "buy or sell, you MUST call place_order separately for each asset."
            ),
            "after": (
                "CRITICAL: record_decision ONLY logs your intent for audit. "
                "Restating the same top_convictions bar after bar while you "
                "hold ~0 shares of those legs is NOT acting on the thesis - "
                "only place_order moves the portfolio toward the target."
            ),
        },
        {
            "label": "§4.3 quantity sizing recipe  ·  rebalance off the gap",
            "before": (
                "Initial-build recipe only:\n\n"
                "  notional_for_asset = cash_available × weight_of_asset\n"
                "  quantity = max(1, floor(notional / approx_price))\n\n"
                "No guidance for subsequent rebalances."
            ),
            "after": (
                "Initial build recipe (above) + new rebalance recipe:\n\n"
                "  current_weight = (shares_held × approx_price) / portfolio_value\n"
                "  gap = target_weight - current_weight\n"
                "  notional_gap = portfolio_value × gap\n"
                "  quantity = max(1, floor(abs(notional_gap) / approx_price))\n"
                "  side = \"buy\" if gap > 0 else \"sell\""
            ),
        },
        {
            "label": "§5 step 4  ·  when to place orders",
            "before": (
                "place_order ZERO OR MORE times. Submit one place_order per asset "
                "you want to buy or sell this bar. If your edge is genuinely flat, "
                "skip the orders and just advance."
            ),
            "after": (
                "place_order CONDITIONAL on weight gaps. For each asset in "
                "top_convictions, compute current_weight and compare to target.\n\n"
                "TRIGGER a rebalance order if any leg has current_weight < "
                "0.5 * target, OR if drift gap > 2%."
            ),
        },
    ],
    "footer": (
        "Untouched across all 3 reflections:  anti-memorization rules-clause  ·  "
        "tool surface  ·  tier definitions  ·  output contract  ·  worked example "
        "shape  ·  sandbox spec.\n"
        "Opus identified these as already working and spent the 1.10x length "
        "budget tightening the order-trigger logic instead."
    ),
}


def build_xml(config: dict | None = None) -> str:
    if config is None:
        config = _CONFIG_QWEN

    cells: list[str] = []
    next_id = [10]

    def nid() -> int:
        i = next_id[0]
        next_id[0] += 1
        return i

    # === Title bar ===
    title_id = nid()
    cells.append(_cell(
        cid=title_id,
        value=config["title"],
        x=40, y=20, w=W - 80, h=56,
        fill=_BG_TITLE, stroke=_DARK,
        font_size=18, bold=True, align="center",
    ))
    subtitle_id = nid()
    cells.append(_cell(
        cid=subtitle_id,
        value=config["subtitle"],
        x=40, y=80, w=W - 80, h=30,
        fill="white", stroke="white",
        font_size=12, align="center",
    ))

    # === Summary cards: total length stats ===
    base_card_id = nid()
    cells.append(_cell(
        cid=base_card_id,
        value=config["baseline_card"],
        x=COL_LEFT_X, y=130, w=COL_LEFT_W, h=70,
        fill=_RED_FILL, stroke=_RED_STROKE,
        font_size=14, bold=True, align="center",
    ))
    final_card_id = nid()
    cells.append(_cell(
        cid=final_card_id,
        value=config["final_card"],
        x=COL_RIGHT_X, y=130, w=COL_RIGHT_W, h=70,
        fill=_GREEN_FILL, stroke=_GREEN_STROKE,
        font_size=14, bold=True, align="center",
    ))
    cells.append(_arrow(cid=nid(), source_id=base_card_id, target_id=final_card_id))

    # === Section header ===
    cells.append(_cell(
        cid=nid(),
        value=config["section_header"],
        x=40, y=225, w=W - 80, h=30,
        fill="white", stroke="white",
        font_size=13, bold=True, align="left",
    ))

    # === Per-change rows. Three rows at fixed y-bands ===
    row_layouts = [
        {"label_y": 275, "card_y": 305, "card_h": 130},
        {"label_y": 460, "card_y": 490, "card_h": 140},
        {"label_y": 655, "card_y": 685, "card_h": 120},
    ]
    for change, layout in zip(config["changes"], row_layouts):
        # Label row.
        cells.append(_cell(
            cid=nid(),
            value=change["label"],
            x=40, y=layout["label_y"], w=W - 80, h=26,
            fill="white", stroke="white",
            font_size=12, bold=True, align="left",
        ))
        # Before card (left, red).
        left_id = nid()
        cells.append(_cell(
            cid=left_id,
            value=change["before"],
            x=COL_LEFT_X, y=layout["card_y"], w=COL_LEFT_W, h=layout["card_h"],
            fill=_RED_FILL, stroke=_RED_STROKE,
            font_size=11, mono=True, align="left", v_align="top",
        ))
        # After card (right, green).
        right_id = nid()
        cells.append(_cell(
            cid=right_id,
            value=change["after"],
            x=COL_RIGHT_X, y=layout["card_y"], w=COL_RIGHT_W, h=layout["card_h"],
            fill=_GREEN_FILL, stroke=_GREEN_STROKE,
            font_size=11, mono=True, align="left", v_align="top",
        ))
        cells.append(_arrow(cid=nid(), source_id=left_id, target_id=right_id))

    # === Footer: untouched ===
    cells.append(_cell(
        cid=nid(),
        value=config["footer"],
        x=40, y=830, w=W - 80, h=70,
        fill="#fafafa", stroke=_GRAY,
        font_size=11, align="center", v_align="middle",
    ))

    body = "\n".join(cells)
    return (
        '<mxfile host="app.diagrams.net" agent="trade-bench-viz">'
        '<diagram name="prompt-evolution" id="prompt-evolution">'
        '<mxGraphModel dx="1280" dy="900" grid="1" gridSize="10" guides="1" '
        'tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" '
        'pageWidth="1280" pageHeight="920" math="0" shadow="0">'
        '<root>'
        '<mxCell id="0"/><mxCell id="1" parent="0"/>'
        f"{body}"
        '</root>'
        '</mxGraphModel>'
        '</diagram>'
        '</mxfile>'
    )


_MODEL_CONFIGS = {
    "qwen-qwen3-32b-groq": _CONFIG_QWEN,
    "zai-glm-5.1-together": _CONFIG_GLM,
}


def render(out: Path, model_id: str = "qwen-qwen3-32b-groq") -> None:
    """Write the .drawio.xml file. The .png is exported manually from draw.io."""
    config = _MODEL_CONFIGS[model_id]
    xml = build_xml(config)
    out_xml = out.with_suffix(".drawio.xml")
    out_xml.write_text(xml, encoding="utf-8")
    print(f"  XML written to {out_xml}")
    print("  Open it in draw.io and export PNG to:", out)


if __name__ == "__main__":
    import sys
    model_id = sys.argv[1] if len(sys.argv) > 1 else "qwen-qwen3-32b-groq"
    suffix = "" if model_id == "qwen-qwen3-32b-groq" else "_glm"
    out = PROJECT_ROOT / "docs" / "figures" / f"prompt_evolution{suffix}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    render(out, model_id=model_id)
