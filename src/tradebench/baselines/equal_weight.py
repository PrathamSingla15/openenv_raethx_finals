"""Equal-weight baseline — 1/N exposure, one-shot allocation, hold-to-end.

On the first decision day, the baseline calls ``view_universe`` to get the
visible asset list, then sizes a position of ``target_exposure / N``
notional in each asset and submits ``place_order`` calls. From day 2
onwards it just records a no-op decision and advances — no rebalancing,
which keeps turnover low (so ``r_turnover`` ≈ 0) and isolates the wealth
component of the reward.

Quantity sizing is integer-shares only (matches the env's
``PlaceOrderInput.quantity: int = Field(gt=0)`` constraint). Asset prices
are unknown to the baseline (no sandbox_exec call), so we fall back to a
nominal reference price for sizing — anchored to the rescale base used by
the real-data baker (``BASE_PRICE = 75``), which keeps the resulting
exposure within ±10% of the target across all three tiers. The baseline
is meant to be a reference, not a tuned strategy.
"""

from __future__ import annotations

from typing import Any

try:
    from ..models import TradeAction  # type: ignore[no-redef]
except (ImportError, ValueError):  # pragma: no cover
    from models import TradeAction  # type: ignore[no-redef]

from ._common import EnvLike, empty_baseline_result

_TARGET_EXPOSURE = 0.90
_NOMINAL_PRICE = 75.0  # rescale base used by scripts/build_real_dataset.py


async def run_equal_weight_baseline(
    env: EnvLike,
    *,
    tier: str = "t1",
    max_steps: int = 1_000,
    target_exposure: float = _TARGET_EXPOSURE,
) -> dict[str, Any]:
    """Drive ``env`` through one equal-weight one-shot episode of ``tier``."""

    reset = await env.reset(task_tier=tier)
    obs = reset.observation if hasattr(reset, "observation") else reset
    if obs.error:
        return empty_baseline_result("equal_weight", tier)

    initial_value = float(obs.portfolio_value)

    universe_result = await env.step(
        TradeAction(action_type="view_universe", payload={}),
    )
    universe_meta = (universe_result.observation.tool_metadata or {}).get(
        "universe",
        [],
    )
    asset_ids = [row["asset_id"] for row in universe_meta if row.get("asset_id")]
    n = len(asset_ids)
    per_asset_notional = (target_exposure * initial_value / n) if n > 0 else 0.0
    per_asset_qty = max(1, int(per_asset_notional / _NOMINAL_PRICE))

    await env.step(
        TradeAction(
            action_type="record_decision",
            payload={
                "regime_label": "equal_weight",
                "edge_summary": f"1/N across {n} assets at ~{target_exposure:.0%} exposure",
                "intended_exposure": f"{target_exposure:.4f}",
                "top_convictions": [
                    {"asset_id": aid, "weight": f"{target_exposure / n:.4f}"}
                    for aid in asset_ids
                ],
                "uncertainty": "medium",
            },
        ),
    )

    for idx, asset_id in enumerate(asset_ids):
        await env.step(
            TradeAction(
                action_type="place_order",
                payload={
                    "client_order_id": f"eqw_init_{idx:03d}",
                    "asset_id": asset_id,
                    "side": "buy",
                    "quantity": per_asset_qty,
                },
            ),
        )

    rewards: list[float] = []
    breakdowns: list[dict[str, float]] = []
    steps = 0

    advance = await env.step(TradeAction(action_type="advance_day", payload={}))
    rewards.append(float(advance.reward or 0.0))
    breakdowns.append(dict(advance.observation.reward_breakdown or {}))
    obs = advance.observation
    steps += 1

    while not obs.done and steps < max_steps:
        await env.step(
            TradeAction(
                action_type="record_decision",
                payload={
                    "regime_label": "equal_weight",
                    "edge_summary": "hold equal-weight allocation",
                    "intended_exposure": f"{target_exposure:.4f}",
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
        "name": "equal_weight",
        "tier": tier,
        "rewards": rewards,
        "reward_breakdowns": breakdowns,
        "final_value": float(obs.portfolio_value),
        "steps": steps,
        "ended_with_error": False,
    }


__all__ = ["run_equal_weight_baseline"]
