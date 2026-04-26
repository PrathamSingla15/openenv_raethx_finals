"""Demo tab — interactive driver against the live TradeBench env.

Synchronous httpx against the local FastAPI surface (the UI is mounted
inside the same process).
"""

from __future__ import annotations

import json
import os
from typing import Any

import gradio as gr
import httpx


# Loopback into the same-process FastAPI; override for standalone runs.
_BASE_URL = os.environ.get("TRADEBENCH_BASE_URL", "http://localhost:8000")
_TIMEOUT = httpx.Timeout(30.0, connect=5.0)


_INTRO_HTML = """
<div class="tb-section-eyebrow">live demo · drives the real env</div>
<h2 class="tb-section-title">Every observation here came from the running
session runtime. No mock layer.</h2>

<p class="tb-section-lead tb-section-lead-mute">
This tab calls the same FastAPI server you are connected to —
<code>/reset</code> and <code>/step</code> round-trip through the OpenEnv
<code>Environment</code> subclass. If you see a connection error during a
cold-start, wait a few seconds and retry.
</p>
"""


_ACTION_TYPES = [
    "view_universe",
    "view_time",
    "view_portfolio",
    "view_orders",
    "view_constraints",
    "view_episode_metrics",
    "record_decision",
    "place_order",
    "cancel_order",
    "advance_day",
    "sandbox_exec",
]

# Payload hints reference aliased asset ids only (e.g. tier_t1_a01); real
# ticker symbols never appear in catalog output.
_PAYLOAD_HINTS: dict[str, dict[str, Any]] = {
    "view_universe": {},
    "view_time": {},
    "view_portfolio": {},
    "view_orders": {},
    "view_constraints": {},
    "view_episode_metrics": {},
    "record_decision": {
        "regime_label": "low_vol_bull",
        "edge_summary": "momentum tilt; mean-reversion fade on extremes",
        "intended_exposure": 0.4,
        "top_convictions": [
            {"asset_id": "tier_t1_a01", "weight": 0.2},
            {"asset_id": "tier_t1_a02", "weight": 0.2},
        ],
        "uncertainty": "medium",
        "reasoning": "Compute 20-bar rolling Sharpe on the visible universe; allocate Kelly half-fraction.",
    },
    "place_order": {
        "client_order_id": "ord-1",
        "asset_id": "tier_t1_a01",
        "side": "buy",
        "quantity": 10,
    },
    "cancel_order": {"client_order_id": "ord-1"},
    "advance_day": {},
    "sandbox_exec": {
        "command": ["python3", "-c", "print('hello from sandbox')"],
        "env": {},
        "timeout_seconds": 5,
    },
}


def _post(path: str, body: dict[str, Any]) -> dict[str, Any]:
    with httpx.Client(base_url=_BASE_URL, timeout=_TIMEOUT) as client:
        resp = client.post(path, json=body)
        resp.raise_for_status()
        return resp.json()


def _get(path: str) -> dict[str, Any]:
    with httpx.Client(base_url=_BASE_URL, timeout=_TIMEOUT) as client:
        resp = client.get(path)
        resp.raise_for_status()
        return resp.json()


def _format_summary(payload: dict[str, Any]) -> str:
    obs = payload.get("observation", {}) or {}
    reward = payload.get("reward", obs.get("reward"))
    done = payload.get("done", obs.get("done", False))
    pieces = [
        f"date={obs.get('current_date', '?')}",
        f"phase={obs.get('phase', '?')}",
        f"bars_remaining={obs.get('bars_remaining', '?')}",
        f"value={obs.get('portfolio_value', 0):,.2f}",
        f"reward={reward if reward is None else f'{float(reward):+.4f}'}",
        f"done={done}",
    ]
    return "  ·  ".join(pieces)


_REWARD_KEYS = (
    "r_wealth",
    "r_sharpe_bonus",
    "r_drawdown",
    "r_turnover",
    "r_concentration",
    "r_rules",
    "r_hack",
)

_REWARD_BOUNDS = {
    "r_wealth": "[unbounded]",
    "r_sharpe_bonus": "[0, 0.20]",
    "r_drawdown": "[-0.30, 0]",
    "r_turnover": "[-0.10, 0]",
    "r_concentration": "[-0.10, 0]",
    "r_rules": "{-1, 0}",
    "r_hack": "{-1, 0}",
}


def _reward_breakdown_html(payload: dict[str, Any]) -> str:
    """Render the live reward_breakdown as 7 animated bars."""

    obs = payload.get("observation", {}) or {}
    breakdown = obs.get("reward_breakdown") or {}
    if not breakdown:
        return (
            '<div class="tb-reward-table" style="opacity: 0.55;">'
            '<div class="tb-reward-row"><div class="tb-reward-name">'
            'reward_breakdown</div><div></div><div class="tb-reward-value">—</div>'
            '<div class="tb-reward-bound">step the env to populate</div></div>'
            "</div>"
        )

    rows: list[str] = []
    for key in _REWARD_KEYS:
        value = float(breakdown.get(key, 0.0))
        # Soft display cap per bar so the secondary terms remain legible.
        cap = 0.2 if key not in ("r_wealth", "r_rules", "r_hack") else 1.0
        if key == "r_wealth":
            cap = max(0.05, abs(value) * 1.5 or 0.05)
        pct = min(100.0, abs(value) / cap * 100.0) if cap else 0
        if value == 0.0:
            pct = 0.5
            direction = "accent"
        elif value < 0:
            direction = "neg"
        elif key in ("r_wealth", "r_sharpe_bonus"):
            direction = "pos"
        else:
            direction = "accent"
        bar_left = "0" if value >= 0 else f"{50 - pct/2:.2f}%"
        rows.append(
            f"<div class='tb-reward-row'>"
            f"<div class='tb-reward-name'>{key}</div>"
            f"<div class='tb-reward-bar-track'>"
            f"<div class='tb-reward-bar-fill {direction}' "
            f"style='left: {bar_left}; width: {pct:.2f}%;'></div></div>"
            f"<div class='tb-reward-value'>{value:+.4f}</div>"
            f"<div class='tb-reward-bound'>{_REWARD_BOUNDS[key]}</div></div>",
        )
    total = float(payload.get("reward") or obs.get("reward") or 0.0)
    rows.append(
        "<div class='tb-reward-row' style='border-top: 1px solid var(--rule-strong); padding-top: 8px;'>"
        "<div class='tb-reward-name'><strong>r_total</strong></div>"
        "<div></div>"
        f"<div class='tb-reward-value'><strong>{total:+.4f}</strong></div>"
        "<div class='tb-reward-bound'>r_wealth + 0.5·Σsec</div></div>",
    )
    return "<div class='tb-reward-table'>" + "".join(rows) + "</div>"


def _hint_for(action_type: str) -> str:
    return json.dumps(_PAYLOAD_HINTS.get(action_type, {}), indent=2)


_RESET_LABEL_HTML = (
    '<div class="tb-section-eyebrow">step 1 · reset an episode</div>'
)
_STEP_LABEL_HTML = (
    '<div class="tb-section-eyebrow">step 2 · step the environment</div>'
)
_BREAKDOWN_LABEL_HTML = (
    '<div class="tb-section-eyebrow">live composite reward</div>'
)
_SAMPLE_LABEL_HTML = (
    '<div class="tb-section-eyebrow">step 3 · run a one-bar sample</div>'
)


def render() -> None:
    gr.HTML(_INTRO_HTML)

    gr.HTML(_RESET_LABEL_HTML)
    with gr.Row():
        with gr.Column(scale=1):
            tier = gr.Dropdown(
                choices=["t1", "train", "test"],
                value="t1",
                label="Task tier",
                info="t1=60 bars (debug), train=252 bars (study packet), test=120 bars (held-out rollout)",
            )
            seed = gr.Number(value=42, label="Seed", precision=0)
            reset_btn = gr.Button("Reset environment", variant="primary")
            reset_status = gr.Textbox(
                label="Status",
                value="Not started.",
                interactive=False,
                lines=2,
            )
        with gr.Column(scale=2):
            reset_obs = gr.JSON(label="Initial observation")

    gr.HTML(_STEP_LABEL_HTML)
    with gr.Row():
        with gr.Column(scale=1):
            action_type = gr.Dropdown(
                choices=_ACTION_TYPES,
                value="view_portfolio",
                label="Action type",
            )
            payload_box = gr.Code(
                value=_hint_for("view_portfolio"),
                language="json",
                label="Payload (JSON)",
                lines=10,
            )
            step_btn = gr.Button("Step", variant="primary")
            step_status = gr.Textbox(
                label="Status",
                value="No step yet.",
                interactive=False,
                lines=2,
            )
        with gr.Column(scale=2):
            step_obs = gr.JSON(label="Step result")

    gr.HTML(_BREAKDOWN_LABEL_HTML)
    breakdown_html = gr.HTML(_reward_breakdown_html({}))

    gr.HTML(_SAMPLE_LABEL_HTML)
    gr.Markdown(
        "Walks through `view_portfolio → view_universe → record_decision → "
        "place_order → advance_day → view_episode_metrics`. Same flow a "
        "training rollout would do for one bar.",
    )
    sample_btn = gr.Button("Run sample episode", variant="primary")
    sample_log = gr.JSON(label="Per-step log", value=[])

    def handle_reset(tier_value: str, seed_value: float):
        try:
            body = {"task_tier": tier_value, "seed": int(seed_value or 0)}
            payload = _post("/reset", body)
            return (
                _format_summary(payload),
                payload,
                _reward_breakdown_html(payload),
            )
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:600]
            return (
                f"HTTP {exc.response.status_code}: {detail}",
                {"error": detail},
                _reward_breakdown_html({}),
            )
        except Exception as exc:  # noqa: BLE001
            return f"Reset failed: {exc!s}", {"error": str(exc)}, _reward_breakdown_html({})

    def handle_step(act_type: str, payload_text: str):
        try:
            try:
                payload = json.loads(payload_text) if payload_text.strip() else {}
            except json.JSONDecodeError as je:
                return f"Invalid JSON payload: {je}", {"error": str(je)}, _reward_breakdown_html({})
            body = {"action": {"action_type": act_type, "payload": payload}}
            response = _post("/step", body)
            return (
                _format_summary(response),
                response,
                _reward_breakdown_html(response),
            )
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:600]
            return (
                f"HTTP {exc.response.status_code}: {detail}",
                {"error": detail},
                _reward_breakdown_html({}),
            )
        except Exception as exc:  # noqa: BLE001
            return f"Step failed: {exc!s}", {"error": str(exc)}, _reward_breakdown_html({})

    def handle_action_change(act_type: str):
        return _hint_for(act_type)

    def handle_sample_episode(tier_value: str):
        log: list[dict[str, Any]] = []
        try:
            reset_resp = _post("/reset", {"task_tier": tier_value, "seed": 42})
            log.append({"step": "reset", "summary": _format_summary(reset_resp)})

            obs_meta = (reset_resp.get("observation") or {}).get("tool_metadata") or {}
            universe = obs_meta.get("universe") or [f"tier_{tier_value}_a01"]
            primary_asset = universe[0] if isinstance(universe, list) and universe else f"tier_{tier_value}_a01"

            scripted = [
                ("view_portfolio", {}),
                ("view_universe", {}),
                (
                    "record_decision",
                    {
                        "regime_label": "neutral",
                        "edge_summary": "demo: equal-weight, no edge claim",
                        "intended_exposure": 0.3,
                        "top_convictions": [],
                        "uncertainty": "high",
                        "reasoning": "Demo run; no signal computed.",
                    },
                ),
                (
                    "place_order",
                    {
                        "client_order_id": "ord-demo-1",
                        "asset_id": primary_asset,
                        "side": "buy",
                        "quantity": 5,
                    },
                ),
                ("advance_day", {}),
                ("view_episode_metrics", {}),
            ]
            for action_type_name, action_payload in scripted:
                resp = _post(
                    "/step",
                    {"action": {"action_type": action_type_name, "payload": action_payload}},
                )
                log.append(
                    {
                        "step": action_type_name,
                        "summary": _format_summary(resp),
                        "tool_output_preview": (
                            (resp.get("observation") or {}).get("tool_output", "")[:200]
                        ),
                    },
                )
        except Exception as exc:  # noqa: BLE001
            log.append({"step": "ERROR", "summary": str(exc)})
        return log

    reset_btn.click(
        fn=handle_reset,
        inputs=[tier, seed],
        outputs=[reset_status, reset_obs, breakdown_html],
    )
    step_btn.click(
        fn=handle_step,
        inputs=[action_type, payload_box],
        outputs=[step_status, step_obs, breakdown_html],
    )
    action_type.change(
        fn=handle_action_change,
        inputs=[action_type],
        outputs=[payload_box],
    )
    sample_btn.click(
        fn=handle_sample_episode,
        inputs=[tier],
        outputs=[sample_log],
    )
