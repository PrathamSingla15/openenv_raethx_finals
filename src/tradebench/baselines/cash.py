"""Cash-only baseline — never holds risky positions.

Each decision day records a zero-exposure decision and immediately advances.
The expected reward path is approximately flat: ``r_wealth ≈ 0`` (cash earns
nothing in this env), no drawdown, no turnover, no concentration. This sets
the absolute lower bound on the improvement-evidence plot.
"""

from __future__ import annotations

from typing import Any

try:
    from ..models import TradeAction  # type: ignore[no-redef]
except (ImportError, ValueError):  # pragma: no cover - dev import fallback
    from models import TradeAction  # type: ignore[no-redef]

from ._common import EnvLike, empty_baseline_result


async def run_cash_baseline(
    env: EnvLike,
    *,
    tier: str = "t1",
    max_steps: int = 1_000,
) -> dict[str, Any]:
    """Drive ``env`` through one cash-only episode of ``tier``.

    Returns a summary dict with per-bar rewards, per-bar 7-component
    breakdowns, and the terminal portfolio value. ``max_steps`` is a hard
    safety cap; it should comfortably exceed the longest tier window
    (T3 ≈ 252 bars, with 2 actions/bar = 504 steps).
    """

    reset = await env.reset(task_tier=tier)
    obs = reset.observation if hasattr(reset, "observation") else reset
    if obs.error:
        return empty_baseline_result("cash", tier)

    rewards: list[float] = []
    breakdowns: list[dict[str, float]] = []
    steps = 0
    while not obs.done and steps < max_steps:
        await env.step(
            TradeAction(
                action_type="record_decision",
                payload={
                    "regime_label": "cash_only",
                    "edge_summary": "no signal; preserve capital",
                    "intended_exposure": "0.0",
                    "top_convictions": [],
                    "uncertainty": "low",
                },
            ),
        )
        result = await env.step(TradeAction(action_type="advance_day", payload={}))
        rewards.append(float(result.reward or 0.0))
        breakdowns.append(dict(result.observation.reward_breakdown or {}))
        obs = result.observation
        steps += 1

    return {
        "name": "cash",
        "tier": tier,
        "rewards": rewards,
        "reward_breakdowns": breakdowns,
        "final_value": float(obs.portfolio_value),
        "steps": steps,
        "ended_with_error": False,
    }


__all__ = ["run_cash_baseline"]
