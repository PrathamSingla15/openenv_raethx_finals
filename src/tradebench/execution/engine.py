"""Daily execution: validate, corporate actions, next-open fills, DayAdvanced."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Literal, cast

import pandas as pd  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field

from tradebench.data.models import CorporateActionKind
from tradebench.data.query import PitQueryService
from tradebench.episodes.models import EpisodeManifest
from tradebench.execution.corporate_actions import (
    DelistingLedgerEffect,
    DividendLedgerEffect,
    SplitLedgerEffect,
    TickerChangeLedgerEffect,
    plan_corporate_action_ledger_effects,
)
from tradebench.execution.costs import (
    CostModel,
    compute_fee_total,
    compute_slippage_bps,
    fill_price_from_open,
)
from tradebench.ledger.events import (
    DayAdvanced,
    DelistingLiquidated,
    DividendApplied,
    LedgerEvent,
    OrderCancelled,
    OrderFilled,
    OrderRejected,
    OrderSubmitted,
    SplitApplied,
    TickerChanged,
)
from tradebench.ledger.projector import project
from tradebench.ledger.state import PortfolioState


class OrderIntent(BaseModel):
    """Agent order to validate at decision close and execute next session open."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    client_order_id: str
    asset_id: str
    side: Literal["buy", "sell"]
    quantity: int = Field(gt=0)


def _coerce_session_date(value: object) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return cast(date, pd.Timestamp(value).date())


def _session_bar_row(
    pit: PitQueryService,
    asset_id: str,
    session_date: date,
) -> pd.Series:
    df = pit.get_bars([asset_id], session_date, lookback_days=1)
    if df.empty:
        msg = f"missing bar for asset_id={asset_id!r} on {session_date}"
        raise ValueError(msg)
    mask = df["session_date"].map(_coerce_session_date) == session_date
    sub = df[mask]
    if sub.empty:
        msg = f"no bar on session_date={session_date} for asset_id={asset_id!r}"
        raise ValueError(msg)
    return sub.iloc[-1]


def _worst_case_buy_reservation(
    model: CostModel,
    *,
    quantity: int,
    decision_close: Decimal,
) -> Decimal:
    """Reserve against the decision close with the slippage cap fully applied."""

    max_px = decision_close * (
        Decimal("1") + model.slippage_bps_cap / Decimal("10000")
    )
    worst_notional = Decimal(quantity) * max_px
    return worst_notional + compute_fee_total(model, fill_notional=worst_notional)


def portfolio_market_value(
    portfolio: PortfolioState,
    per_share_close: dict[str, Decimal],
) -> Decimal:
    """Mark long positions at supplied closes; cash included."""

    total = portfolio.cash
    for aid, pos in portfolio.positions_by_asset.items():
        px = per_share_close.get(aid)
        if px is None:
            msg = f"missing close for position asset_id={aid!r}"
            raise ValueError(msg)
        total += Decimal(pos.shares) * px
    return total


def unrealized_from_marks(
    portfolio: PortfolioState,
    per_share_close: dict[str, Decimal],
) -> Decimal:
    """Unrealized PnL vs remaining cost basis using session closes."""

    unreal = Decimal("0")
    for aid, pos in portfolio.positions_by_asset.items():
        px = per_share_close[aid]
        mtm = Decimal(pos.shares) * px
        unreal += mtm - pos.cost_basis_total
    return unreal


def advance_trading_session(
    *,
    pit: PitQueryService,
    manifest: EpisodeManifest,
    episode_id: str,
    decision_date: date,
    execution_date: date,
    prior_events: Sequence[LedgerEvent],
    new_orders: Sequence[OrderIntent],
    cost_model: CostModel,
    base_event_time: datetime,
) -> list[LedgerEvent]:
    """
    One trading step from **decision_date** ``t`` to **execution_date** ``t+1``.

    Event order:

    1. Validate ``new_orders`` using **decision_date** closes and PIT universe; emit
       ``OrderSubmitted`` or ``OrderRejected``.
    2. If a **stock split** is effective on ``execution_date`` for an asset, cancel any
       still-open orders on that asset (``OrderCancelled``) before the split is applied.
    3. Apply corporate actions effective on ``execution_date``.
    4. Fill remaining open orders at **execution_date** ``open`` with slippage and fees.
    5. Emit ``DayAdvanced`` for ``execution_date`` with MTM at **execution_date** close.

    Fills use the **regular-way open** on ``execution_date``, adjusted by slippage.
    Trailing dollar volume for slippage is the **decision_date** bar ``dollar_volume``.
    """

    events: list[LedgerEvent] = []
    if base_event_time.tzinfo is None:
        tick = base_event_time.replace(tzinfo=UTC)
    else:
        tick = base_event_time
    seq = 0
    id_seq = 0

    def _next_time() -> datetime:
        nonlocal seq, tick
        t = tick + timedelta(microseconds=seq)
        seq += 1
        return t

    def _eid(suffix: str) -> str:
        nonlocal id_seq
        id_seq += 1
        return f"{episode_id}:{execution_date.isoformat()}:{id_seq:05d}:{suffix}"

    universe = pit.load_universe(decision_date, manifest)
    universe_set = set(universe)
    symbol_map = pit.get_symbol_to_asset_id(decision_date, universe)

    corp_rows = pit.get_corporate_actions(universe, execution_date)
    split_assets = {
        r.asset_id for r in corp_rows if r.action_type == CorporateActionKind.SPLIT
    }

    for intent in new_orders:
        t = _next_time()
        rolling = project([*prior_events, *events])
        if intent.asset_id not in universe_set:
            events.append(
                OrderRejected(
                    event_id=_eid(f"rej_{intent.client_order_id}"),
                    episode_id=episode_id,
                    event_time=t,
                    client_order_id=intent.client_order_id,
                ),
            )
            continue

        row = _session_bar_row(pit, intent.asset_id, decision_date)
        if intent.side == "sell":
            pos = rolling.positions_by_asset.get(intent.asset_id)
            if pos is None or pos.shares < intent.quantity:
                events.append(
                    OrderRejected(
                        event_id=_eid(f"rej_{intent.client_order_id}"),
                        episode_id=episode_id,
                        event_time=t,
                        client_order_id=intent.client_order_id,
                    ),
                )
                continue
            events.append(
                OrderSubmitted(
                    event_id=_eid(f"sub_{intent.client_order_id}"),
                    episode_id=episode_id,
                    event_time=t,
                    client_order_id=intent.client_order_id,
                    asset_id=intent.asset_id,
                    side="sell",
                    quantity=intent.quantity,
                    cash_reserved=Decimal("0"),
                ),
            )
            continue

        close_d = Decimal(str(row["close"]))
        reserve = _worst_case_buy_reservation(
            cost_model,
            quantity=intent.quantity,
            decision_close=close_d,
        )
        if rolling.cash < reserve:
            events.append(
                OrderRejected(
                    event_id=_eid(f"rej_{intent.client_order_id}"),
                    episode_id=episode_id,
                    event_time=t,
                    client_order_id=intent.client_order_id,
                ),
            )
            continue
        events.append(
            OrderSubmitted(
                event_id=_eid(f"sub_{intent.client_order_id}"),
                episode_id=episode_id,
                event_time=t,
                client_order_id=intent.client_order_id,
                asset_id=intent.asset_id,
                side="buy",
                quantity=intent.quantity,
                cash_reserved=reserve,
            ),
        )

    state_after_submit = project([*prior_events, *events])

    for oid, oo in list(state_after_submit.open_orders.items()):
        if oo.asset_id in split_assets:
            events.append(
                OrderCancelled(
                    event_id=_eid(f"cxl_split_{oid}"),
                    episode_id=episode_id,
                    event_time=_next_time(),
                    client_order_id=oid,
                ),
            )

    state_after_cxls = project([*prior_events, *events])

    def _fees(notional: Decimal) -> Decimal:
        return compute_fee_total(cost_model, fill_notional=notional)

    planned = plan_corporate_action_ledger_effects(
        corp_rows,
        portfolio=state_after_cxls,
        pit=pit,
        symbol_to_asset_id=symbol_map,
        fee_for_notional=_fees,
    )

    for effect in planned:
        et = _next_time()
        if isinstance(effect, DividendLedgerEffect):
            events.append(
                DividendApplied(
                    event_id=_eid("div"),
                    episode_id=episode_id,
                    event_time=et,
                    asset_id=effect.asset_id,
                    cash_credited=effect.cash_credited,
                ),
            )
        elif isinstance(effect, SplitLedgerEffect):
            events.append(
                SplitApplied(
                    event_id=_eid("spl"),
                    episode_id=episode_id,
                    event_time=et,
                    asset_id=effect.asset_id,
                    ratio=effect.ratio,
                ),
            )
        elif isinstance(effect, TickerChangeLedgerEffect):
            events.append(
                TickerChanged(
                    event_id=_eid("tkr"),
                    episode_id=episode_id,
                    event_time=et,
                    from_asset_id=effect.from_asset_id,
                    to_asset_id=effect.to_asset_id,
                ),
            )
        elif isinstance(effect, DelistingLedgerEffect):
            events.append(
                DelistingLiquidated(
                    event_id=_eid("dlst"),
                    episode_id=episode_id,
                    event_time=et,
                    asset_id=effect.asset_id,
                    quantity_liquidated=effect.quantity_liquidated,
                    cash_proceeds=effect.cash_proceeds,
                    fees=effect.fees,
                ),
            )
        else:
            msg = f"unsupported corporate action effect: {type(effect).__name__}"
            raise TypeError(msg)

    state_after_corp = project([*prior_events, *events])

    for oid, oo in sorted(state_after_corp.open_orders.items()):
        bar_exec = _session_bar_row(pit, oo.asset_id, execution_date)
        open_px = Decimal(str(bar_exec["open"]))
        bar_dec = _session_bar_row(pit, oo.asset_id, decision_date)
        trail_dv = Decimal(str(bar_dec["dollar_volume"]))
        notional = Decimal(oo.quantity) * open_px
        slip_bps = compute_slippage_bps(
            cost_model,
            order_notional=notional,
            trailing_dollar_volume=trail_dv,
        )
        avg_px = fill_price_from_open(open_px, oo.side, slip_bps)
        fill_notional = Decimal(oo.quantity) * avg_px
        fees = _fees(fill_notional)
        events.append(
            OrderFilled(
                event_id=_eid(f"fill_{oid}"),
                episode_id=episode_id,
                event_time=_next_time(),
                client_order_id=oid,
                asset_id=oo.asset_id,
                side=oo.side,
                quantity=oo.quantity,
                avg_fill_price=avg_px,
                fees=fees,
            ),
        )

    state_final = project([*prior_events, *events])

    closes: dict[str, Decimal] = {}
    for aid in sorted(state_final.positions_by_asset.keys()):
        row_c = _session_bar_row(pit, aid, execution_date)
        closes[aid] = Decimal(str(row_c["close"]))

    mtm = portfolio_market_value(state_final, closes)
    unreal = unrealized_from_marks(state_final, closes)

    events.append(
        DayAdvanced(
            event_id=_eid("day"),
            episode_id=episode_id,
            event_time=_next_time(),
            session_date=execution_date,
            portfolio_market_value=mtm,
            unrealized_pnl=unreal,
        ),
    )

    return events
