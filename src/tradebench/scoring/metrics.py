"""Pure episode scoring metrics derived from explicit PIT data and ledger events.

The primary score is cumulative log wealth:

``log(final_portfolio_value / initial_cash)``

Secondary metrics intentionally use simple, explicit formulas:

- Step reward: ``log(V_t1 / V_t0)``
- Max drawdown: worst peak-to-trough decline on the portfolio value path
- Realized volatility: annualized sample standard deviation of step rewards
- Sharpe: annualized mean(step rewards) / sample std(step rewards), risk-free = 0
- Sortino: annualized mean(step rewards) / downside deviation(step rewards)
- Turnover: sum over sessions of ``traded_notional / previous_portfolio_value``
- Concentration: mean end-of-day Herfindahl index of risky sleeve weights
- Cost drag: ``total_explicit_costs / initial_cash``
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import date, datetime
from decimal import Decimal
from typing import cast

import pandas as pd  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field

from tradebench.data.query import PitQueryService
from tradebench.episodes.models import EpisodeManifest
from tradebench.ledger.events import (
    DayAdvanced,
    DelistingLiquidated,
    LedgerEvent,
    OrderFilled,
)
from tradebench.ledger.projector import project
from tradebench.ledger.state import PortfolioState

TRADING_DAYS_PER_YEAR = 252


class PositionExposure(BaseModel):
    """Realized end-of-day position weight for one asset."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    asset_id: str
    close_price: Decimal
    market_value: Decimal
    portfolio_weight: Decimal


class SessionMetrics(BaseModel):
    """Per-session performance slice anchored on a ``DayAdvanced`` event."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_time: datetime
    session_date: date
    portfolio_value: Decimal
    step_reward: float
    cumulative_log_wealth: float
    gross_exposure: Decimal
    net_exposure: Decimal
    traded_notional: Decimal
    transaction_costs: Decimal
    turnover: float
    concentration: float
    positions: list[PositionExposure] = Field(default_factory=list)


class EpisodeMetricsSummary(BaseModel):
    """Deterministic episode-level performance summary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    initial_value: Decimal
    final_value: Decimal
    score: float
    cumulative_log_wealth: float
    max_drawdown: float
    realized_volatility: float
    sharpe: float
    sortino: float
    turnover: float
    concentration: float
    total_costs: Decimal
    cost_drag: float
    step_rewards: list[float] = Field(default_factory=list)
    sessions: list[SessionMetrics] = Field(default_factory=list)


def step_log_reward(value_t0: Decimal, value_t1: Decimal) -> float:
    """Return the economic step reward ``log(V_t1 / V_t0)``."""

    if value_t0 <= 0 or value_t1 <= 0:
        msg = "portfolio values must be positive to compute log rewards"
        raise ValueError(msg)
    return math.log(float(value_t1 / value_t0))


def cumulative_log_wealth(initial_value: Decimal, current_value: Decimal) -> float:
    """Return cumulative log wealth relative to the episode start."""

    return step_log_reward(initial_value, current_value)


def max_drawdown(values: Sequence[Decimal]) -> float:
    """Worst peak-to-trough drawdown on the supplied portfolio value path."""

    if not values:
        return 0.0
    peak = values[0]
    worst = Decimal("0")
    for value in values:
        if value > peak:
            peak = value
        if peak > 0:
            drawdown = (peak - value) / peak
            if drawdown > worst:
                worst = drawdown
    return float(worst)


def realized_volatility(
    step_rewards: Sequence[float],
    *,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualized sample standard deviation of step rewards."""

    if len(step_rewards) < 2:
        return 0.0
    mean_reward = sum(step_rewards) / len(step_rewards)
    sample_var = sum((reward - mean_reward) ** 2 for reward in step_rewards) / (
        len(step_rewards) - 1
    )
    return math.sqrt(sample_var) * math.sqrt(periods_per_year)


def sharpe_ratio(
    step_rewards: Sequence[float],
    *,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualized Sharpe ratio on step rewards with zero risk-free rate."""

    if len(step_rewards) < 2:
        return 0.0
    mean_reward = sum(step_rewards) / len(step_rewards)
    sample_var = sum((reward - mean_reward) ** 2 for reward in step_rewards) / (
        len(step_rewards) - 1
    )
    if sample_var <= 0:
        return 0.0
    return mean_reward / math.sqrt(sample_var) * math.sqrt(periods_per_year)


def sortino_ratio(
    step_rewards: Sequence[float],
    *,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualized Sortino ratio using downside deviation on negative rewards only."""

    if not step_rewards:
        return 0.0
    mean_reward = sum(step_rewards) / len(step_rewards)
    downside_var = sum(min(reward, 0.0) ** 2 for reward in step_rewards) / len(
        step_rewards,
    )
    if downside_var <= 0:
        return 0.0
    return mean_reward / math.sqrt(downside_var) * math.sqrt(periods_per_year)


def turnover(traded_notional: Decimal, previous_value: Decimal) -> float:
    """Simple session turnover: gross traded notional divided by lagged wealth."""

    if previous_value <= 0:
        return 0.0
    return float(traded_notional / previous_value)


def concentration(weights: Sequence[Decimal]) -> float:
    """Herfindahl concentration of the risky sleeve, ignoring cash."""

    gross = sum(abs(weight) for weight in weights)
    if gross <= 0:
        return 0.0
    hhi = Decimal("0")
    for weight in weights:
        sleeve_weight = abs(weight) / gross
        hhi += sleeve_weight * sleeve_weight
    return float(hhi)


def cost_drag(total_costs: Decimal, initial_value: Decimal) -> float:
    """Explicit cost drag as a fraction of initial capital."""

    if initial_value <= 0:
        return 0.0
    return float(total_costs / initial_value)


def portfolio_marks_at_close(
    *,
    pit: PitQueryService,
    portfolio: PortfolioState,
    session_date: date,
) -> dict[str, Decimal]:
    """Return close marks for every held asset on the supplied session date."""

    asset_ids = sorted(portfolio.positions_by_asset)
    if not asset_ids:
        return {}

    frame = pit.get_bars(asset_ids, session_date, lookback_days=1)
    marks: dict[str, Decimal] = {}
    for asset_id in asset_ids:
        rows = frame[
            (frame["asset_id"] == asset_id)
            & (frame["session_date"].map(_coerce_session_date) == session_date)
        ]
        if rows.empty:
            msg = (
                f"missing close for position asset_id={asset_id!r} "
                f"on {session_date.isoformat()}"
            )
            raise ValueError(msg)
        marks[asset_id] = Decimal(str(rows.iloc[-1]["close"]))
    return marks


def score_episode(
    *,
    manifest: EpisodeManifest,
    pit: PitQueryService,
    events: Sequence[LedgerEvent],
) -> EpisodeMetricsSummary:
    """Build a deterministic episode summary from explicit inputs only."""

    sessions = _build_session_metrics(manifest=manifest, pit=pit, events=events)
    if sessions:
        final_value = sessions[-1].portfolio_value
    else:
        final_value = _fallback_final_value(manifest=manifest, events=events)

    step_rewards = [session.step_reward for session in sessions]
    total_costs = sum(
        (session.transaction_costs for session in sessions),
        start=Decimal("0"),
    )
    value_path = [
        manifest.initial_cash,
        *[session.portfolio_value for session in sessions],
    ]
    score = cumulative_log_wealth(manifest.initial_cash, final_value)
    mean_concentration = 0.0
    if sessions:
        mean_concentration = sum(session.concentration for session in sessions) / len(
            sessions,
        )

    return EpisodeMetricsSummary(
        initial_value=manifest.initial_cash,
        final_value=final_value,
        score=score,
        cumulative_log_wealth=score,
        max_drawdown=max_drawdown(value_path),
        realized_volatility=realized_volatility(step_rewards),
        sharpe=sharpe_ratio(step_rewards),
        sortino=sortino_ratio(step_rewards),
        turnover=sum(session.turnover for session in sessions),
        concentration=mean_concentration,
        total_costs=total_costs,
        cost_drag=cost_drag(total_costs, manifest.initial_cash),
        step_rewards=step_rewards,
        sessions=sessions,
    )


def _build_session_metrics(
    *,
    manifest: EpisodeManifest,
    pit: PitQueryService,
    events: Sequence[LedgerEvent],
) -> list[SessionMetrics]:
    sessions: list[SessionMetrics] = []
    previous_value = manifest.initial_cash
    previous_day_index = -1

    for index, event in enumerate(events):
        if not isinstance(event, DayAdvanced):
            continue

        prefix_events = events[: index + 1]
        state = project(prefix_events)
        marks = portfolio_marks_at_close(
            pit=pit,
            portfolio=state,
            session_date=event.session_date,
        )
        portfolio_value = _portfolio_value_for_session(
            event=event,
            state=state,
            marks=marks,
        )
        positions = _positions_for_session(
            state=state,
            marks=marks,
            portfolio_value=portfolio_value,
        )
        session_events = events[previous_day_index + 1 : index + 1]
        traded_notional = _traded_notional(session_events)
        session_costs = _explicit_costs(session_events)
        weights = [position.portfolio_weight for position in positions]

        session_metrics = SessionMetrics(
            event_time=event.event_time,
            session_date=event.session_date,
            portfolio_value=portfolio_value,
            step_reward=step_log_reward(previous_value, portfolio_value),
            cumulative_log_wealth=cumulative_log_wealth(
                manifest.initial_cash,
                portfolio_value,
            ),
            gross_exposure=sum(
                (abs(position.portfolio_weight) for position in positions),
                start=Decimal("0"),
            ),
            net_exposure=sum(
                (position.portfolio_weight for position in positions),
                start=Decimal("0"),
            ),
            traded_notional=traded_notional,
            transaction_costs=session_costs,
            turnover=turnover(traded_notional, previous_value),
            concentration=concentration(weights),
            positions=positions,
        )
        sessions.append(session_metrics)

        previous_value = portfolio_value
        previous_day_index = index

    return sessions


def _portfolio_value_for_session(
    *,
    event: DayAdvanced,
    state: PortfolioState,
    marks: dict[str, Decimal],
) -> Decimal:
    if event.portfolio_market_value is not None:
        return event.portfolio_market_value
    total = state.cash
    for asset_id, position in state.positions_by_asset.items():
        close_price = marks.get(asset_id)
        if close_price is None:
            msg = f"missing close for asset_id={asset_id!r}"
            raise ValueError(msg)
        total += Decimal(position.shares) * close_price
    return total


def _positions_for_session(
    *,
    state: PortfolioState,
    marks: dict[str, Decimal],
    portfolio_value: Decimal,
) -> list[PositionExposure]:
    if portfolio_value <= 0:
        return []

    positions: list[PositionExposure] = []
    for asset_id in sorted(state.positions_by_asset):
        position = state.positions_by_asset[asset_id]
        close_price = marks[asset_id]
        market_value = Decimal(position.shares) * close_price
        positions.append(
            PositionExposure(
                asset_id=asset_id,
                close_price=close_price,
                market_value=market_value,
                portfolio_weight=market_value / portfolio_value,
            ),
        )
    return positions


def _traded_notional(events: Sequence[LedgerEvent]) -> Decimal:
    total = Decimal("0")
    for event in events:
        if isinstance(event, OrderFilled):
            total += Decimal(event.quantity) * event.avg_fill_price
        elif isinstance(event, DelistingLiquidated):
            total += event.cash_proceeds + event.fees
    return total


def _explicit_costs(events: Sequence[LedgerEvent]) -> Decimal:
    total = Decimal("0")
    for event in events:
        if isinstance(event, OrderFilled):
            total += event.fees
        elif isinstance(event, DelistingLiquidated):
            total += event.fees
    return total


def _fallback_final_value(
    *,
    manifest: EpisodeManifest,
    events: Sequence[LedgerEvent],
) -> Decimal:
    state = project(events)
    if state.last_valuation is not None:
        return state.last_valuation
    if state.cash > 0:
        return state.cash
    return manifest.initial_cash


def _coerce_session_date(value: object) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return cast(date, pd.Timestamp(value).date())


__all__ = [
    "EpisodeMetricsSummary",
    "PositionExposure",
    "SessionMetrics",
    "concentration",
    "cost_drag",
    "cumulative_log_wealth",
    "max_drawdown",
    "portfolio_marks_at_close",
    "realized_volatility",
    "score_episode",
    "sharpe_ratio",
    "sortino_ratio",
    "step_log_reward",
    "turnover",
]
