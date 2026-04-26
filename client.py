"""OpenEnv client for the TradeBench environment.

Speaks the TradeBench wire protocol defined in ``models.py``:
``_step_payload`` serializes a ``TradeAction``, ``_parse_result`` rebuilds
a ``TradeObservation``, ``_parse_state`` rebuilds a ``TradeState``.
"""

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

try:
    from .models import TradeAction, TradeObservation, TradeState
except ImportError:
    from models import TradeAction, TradeObservation, TradeState  # type: ignore[no-redef]


class TradeBenchEnv(EnvClient[TradeAction, TradeObservation, TradeState]):
    """Client for the TradeBench long-horizon trading environment.

    Example:
        >>> async with TradeBenchEnv(base_url="http://localhost:8000") as env:
        ...     result = await env.reset(task_tier="t1")
        ...     result = await env.step(TradeAction(action_type="view_portfolio"))
        ...     result = await env.step(TradeAction(
        ...         action_type="place_order",
        ...         payload={
        ...             "client_order_id": "ord-1",
        ...             "asset_id": "tb_sample_liquid",
        ...             "side": "buy",
        ...             "quantity": 10,
        ...         },
        ...     ))
        ...     result = await env.step(TradeAction(action_type="advance_day"))
    """

    def _step_payload(self, action: TradeAction) -> Dict[str, Any]:
        return {
            "action_type": action.action_type,
            "payload": action.payload,
        }

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[TradeObservation]:
        obs_data = payload.get("observation", {})
        observation = TradeObservation(
            done=payload.get("done", obs_data.get("done", False)),
            reward=payload.get("reward", obs_data.get("reward")),
            metadata=obs_data.get("metadata", {}),
            task_tier=obs_data.get("task_tier", "t1"),
            current_date=obs_data.get("current_date", ""),
            bars_remaining=obs_data.get("bars_remaining", 0),
            phase=obs_data.get("phase", ""),
            tool_output=obs_data.get("tool_output", ""),
            tool_metadata=obs_data.get("tool_metadata", {}),
            reward_breakdown=obs_data.get("reward_breakdown", {}),
            portfolio_value=obs_data.get("portfolio_value", 0.0),
            cash=obs_data.get("cash", 0.0),
            positions=obs_data.get("positions", {}),
            available_actions=obs_data.get("available_actions", []),
            violations=obs_data.get("violations", []),
            error=obs_data.get("error"),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> TradeState:
        return TradeState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_tier=payload.get("task_tier", "t1"),
            current_date=payload.get("current_date", ""),
            initial_cash=payload.get("initial_cash", 100_000.0),
            bars_total=payload.get("bars_total", 0),
            bars_remaining=payload.get("bars_remaining", 0),
            cumulative_reward=payload.get("cumulative_reward", 0.0),
            cumulative_log_wealth=payload.get("cumulative_log_wealth", 0.0),
            violations_count=payload.get("violations_count", 0),
            terminated_on_violation=payload.get("terminated_on_violation", False),
        )
