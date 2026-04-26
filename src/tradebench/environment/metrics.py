"""Ledger-derived metric adapters for the environment surface."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from decimal import Decimal

from pydantic import BaseModel, ConfigDict

from tradebench.data.query import PitQueryService
from tradebench.episodes.models import EpisodeManifest
from tradebench.execution.engine import portfolio_market_value
from tradebench.ledger.events import (
    DelistingLiquidated,
    LedgerEvent,
    OrderFilled,
)
from tradebench.ledger.projector import project
from tradebench.ledger.state import PortfolioState
from tradebench.scoring.metrics import (
    concentration,
    cumulative_log_wealth,
    portfolio_marks_at_close,
)


class EpisodeMetricsSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    current_session_date: date
    next_session_date: date | None
    portfolio_value: str
    score: float
    realized_pnl: str
    unrealized_pnl: str
    cash: str
    positions_count: int
    open_orders_count: int
    queued_orders_count: int
    event_count: int


def build_episode_metrics_snapshot(
    *,
    pit: PitQueryService,
    manifest: EpisodeManifest,
    current_date: date,
    next_session_date: date | None,
    events: list[LedgerEvent],
    queued_orders_count: int,
) -> EpisodeMetricsSnapshot:
    state = project(events)
    marks = portfolio_marks_at_close(
        pit=pit,
        portfolio=state,
        session_date=current_date,
    )
    portfolio_value = portfolio_market_value(state, marks)
    score = cumulative_log_wealth(manifest.initial_cash, portfolio_value)
    return EpisodeMetricsSnapshot(
        current_session_date=current_date,
        next_session_date=next_session_date,
        portfolio_value=str(portfolio_value),
        score=score,
        realized_pnl=str(state.realized_pnl),
        unrealized_pnl=str(state.unrealized_pnl),
        cash=str(state.cash),
        positions_count=len(state.positions_by_asset),
        open_orders_count=len(state.open_orders),
        queued_orders_count=queued_orders_count,
        event_count=max(len(events) - 1, 0),
    )


def reward_delta(*, previous_score: float, current: EpisodeMetricsSnapshot) -> float:
    return current.score - previous_score


def compute_traded_notional(step_events: Sequence[LedgerEvent]) -> Decimal:
    """Gross traded notional within a single advance_day step.

    Sums absolute fill notional from ``OrderFilled`` events plus forced-sell
    proceeds (and explicit fees) from ``DelistingLiquidated``. Mirrors the
    convention in :mod:`tradebench.scoring.metrics`.
    """

    total = Decimal("0")
    for event in step_events:
        if isinstance(event, OrderFilled):
            total += Decimal(event.quantity) * event.avg_fill_price
        elif isinstance(event, DelistingLiquidated):
            total += event.cash_proceeds + event.fees
    return total


def compute_realized_hhi(
    *,
    state: PortfolioState,
    marks: dict[str, Decimal],
    portfolio_value: Decimal,
) -> float:
    """End-of-day Herfindahl of the risky sleeve (cash excluded).

    Returns 0 when the risky sleeve is empty or the portfolio is non-positive.
    The :func:`tradebench.scoring.metrics.concentration` helper renormalizes
    weights internally, so passing ``mv / portfolio_value`` (which leaves a
    cash slack < 1) produces the same value as passing risky-sleeve-only
    weights — both collapse to the within-sleeve HHI.
    """

    if portfolio_value <= 0 or not state.positions_by_asset:
        return 0.0
    weights: list[Decimal] = []
    for asset_id, position in state.positions_by_asset.items():
        mv = Decimal(position.shares) * marks[asset_id]
        weights.append(mv / portfolio_value)
    return concentration(weights)


__all__ = [
    "EpisodeMetricsSnapshot",
    "build_episode_metrics_snapshot",
    "compute_realized_hhi",
    "compute_traded_notional",
    "portfolio_marks_at_close",
    "reward_delta",
]
