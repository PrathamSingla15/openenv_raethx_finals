"""
Pydantic models for the TradeBench environment.

Defines the Action, Observation, and State types exchanged between
the OpenEnv client and the FastAPI server. These are the canonical
typed contracts referenced by models.py / client.py / server/.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


ActionType = Literal[
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


class TradeAction(Action):
    """Agent action in the TradeBench environment.

    The `action_type` selects the tool; `payload` carries its arguments.
    Payload shapes match the frozen Pydantic input classes in
    ``tradebench.environment.tools`` — server-side dispatcher parses
    payload into the right input model.
    """

    action_type: ActionType = Field(
        default="view_portfolio",
        description=(
            "Tool to invoke. One of: view_universe, view_time, view_portfolio, "
            "view_orders, view_constraints, view_episode_metrics, record_decision, "
            "place_order, cancel_order, advance_day, sandbox_exec."
        ),
    )
    payload: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Tool-specific arguments. Empty for view_* and advance_day. "
            "place_order: {client_order_id, asset_id, side, quantity}. "
            "cancel_order: {client_order_id}. "
            "record_decision: {regime_label, edge_summary, intended_exposure, "
            "top_convictions, uncertainty}. "
            "sandbox_exec: {command, env, timeout_seconds}."
        ),
    )


class RewardBreakdown(Observation):
    """Per-component reward decomposition surfaced each step.

    Not an Observation subclass at the wire level — embedded as a dict inside
    ``TradeObservation.reward_breakdown``. This class documents the shape.
    """

    r_wealth: float = 0.0
    r_sharpe_bonus: float = 0.0
    r_drawdown: float = 0.0
    r_turnover: float = 0.0
    r_concentration: float = 0.0
    r_rules: float = 0.0
    r_hack: float = 0.0


class TradeObservation(Observation):
    """Environment response after each action.

    Inherits ``done: bool``, ``reward: float | None``, ``metadata: dict``.
    """

    task_tier: str = Field(default="t1", description="t1 | t2 | t3")
    current_date: str = Field(
        default="",
        description="Simulated current trading date (YYYY-MM-DD). Data past this "
        "is not available to the agent.",
    )
    bars_remaining: int = Field(default=0, description="Trading days left in episode")
    phase: str = Field(
        default="",
        description="Lifecycle phase: setup | trading | settling | done",
    )

    tool_output: str = Field(
        default="",
        description="Human-readable result of the invoked tool (portfolio snapshot, "
        "fills, errors, sandbox stdout, etc.)",
    )
    tool_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured tool output fields (portfolio state, fills, etc.)",
    )

    reward_breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-component reward: r_wealth, r_sharpe_bonus, r_drawdown, "
        "r_turnover, r_concentration, r_rules, r_hack. Sum (with secondary "
        "weighted 0.5x) equals the scalar reward.",
    )

    portfolio_value: float = Field(default=0.0, description="Net portfolio value in USD")
    cash: float = Field(default=0.0, description="Available cash in USD")
    positions: Dict[str, float] = Field(
        default_factory=dict,
        description="Current ticker → quantity map",
    )

    available_actions: List[str] = Field(
        default_factory=list,
        description="Action types legal from the current state",
    )
    violations: List[str] = Field(
        default_factory=list,
        description="Anti-hack / rules-clause violations emitted this step",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if the action was rejected or failed",
    )


class TradeState(State):
    """Episode metadata read via the /state endpoint.

    Inherits ``episode_id: str`` and ``step_count: int``.
    """

    task_tier: Literal["t1", "train", "test"] = Field(
        default="t1",
        description="Difficulty tier of the active episode",
    )
    current_date: str = Field(default="", description="Simulated trading date")
    initial_cash: float = Field(default=100_000.0, description="Starting bankroll")
    bars_total: int = Field(default=0, description="Total bars in the episode")
    bars_remaining: int = Field(default=0, description="Bars left until episode end")
    cumulative_reward: float = Field(
        default=0.0,
        description="Sum of composite step rewards so far (log-wealth + bounded "
        "regularizer terms)",
    )
    cumulative_log_wealth: float = Field(
        default=0.0,
        description="Primary log-wealth signal: log(V_t/V_0)",
    )
    violations_count: int = Field(
        default=0,
        description="Anti-hack / rules-clause violations accumulated this episode",
    )
    terminated_on_violation: bool = Field(
        default=False,
        description="True if episode ended early due to a rules or hack violation",
    )


__all__ = [
    "ActionType",
    "TradeAction",
    "TradeObservation",
    "TradeState",
    "RewardBreakdown",
]
