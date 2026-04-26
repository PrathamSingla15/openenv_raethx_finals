"""Corporate action planning for ledger events (fail-closed on unsupported kinds)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import ROUND_DOWN, Decimal

from tradebench.data.models import CorporateActionKind, CorporateActionRow
from tradebench.data.query import PitQueryService
from tradebench.ledger.state import PortfolioState


class UnsupportedCorporateActionError(ValueError):
    """Raised when a corporate action kind is not implemented."""


@dataclass(frozen=True, slots=True)
class DividendLedgerEffect:
    asset_id: str
    cash_credited: Decimal


@dataclass(frozen=True, slots=True)
class SplitLedgerEffect:
    asset_id: str
    ratio: Decimal


@dataclass(frozen=True, slots=True)
class TickerChangeLedgerEffect:
    from_asset_id: str
    to_asset_id: str


@dataclass(frozen=True, slots=True)
class DelistingLedgerEffect:
    asset_id: str
    quantity_liquidated: int
    cash_proceeds: Decimal
    fees: Decimal


type CorporateActionLedgerEffect = (
    DividendLedgerEffect
    | SplitLedgerEffect
    | TickerChangeLedgerEffect
    | DelistingLedgerEffect
)


def _split_ratio(row: CorporateActionRow) -> Decimal:
    if row.split_from is None or row.split_to is None:
        msg = f"split action missing split_from/split_to for asset_id={row.asset_id!r}"
        raise ValueError(msg)
    if row.split_from <= 0:
        msg = "split_from must be positive"
        raise ValueError(msg)
    return row.split_to / row.split_from


def _corp_action_sort_key(row: CorporateActionRow) -> tuple[int, str, str]:
    """Stable priority: delisting, ticker change, split, dividend, then asset_id."""

    priority = {
        CorporateActionKind.DELISTING: 0,
        CorporateActionKind.TICKER_CHANGE: 1,
        CorporateActionKind.SPLIT: 2,
        CorporateActionKind.CASH_DIVIDEND: 3,
    }.get(row.action_type, 99)
    return (priority, row.asset_id, row.action_type.value)


def last_close_before_effective(
    pit: PitQueryService,
    asset_id: str,
    effective_date: date,
    *,
    lookback_days: int = 600,
) -> Decimal:
    """Liquidate at the last available daily close strictly before effective_date."""

    end = effective_date
    df = pit.get_bars([asset_id], end_date=end, lookback_days=lookback_days)
    if df.empty:
        msg = f"no bars to price delisting for asset_id={asset_id!r}"
        raise ValueError(msg)
    sub = df[df["session_date"] < effective_date]
    if sub.empty:
        msg = (
            f"no session strictly before effective_date={effective_date} "
            f"for {asset_id!r}"
        )
        raise ValueError(msg)
    last = sub.iloc[-1]
    return Decimal(str(last["close"]))


def resolve_ticker_successor_asset_id(
    row: CorporateActionRow,
    symbol_to_asset_id: dict[str, str],
) -> str:
    if row.new_symbol is None or not row.new_symbol:
        msg = "ticker_change missing new_symbol"
        raise ValueError(msg)
    try:
        return symbol_to_asset_id[row.new_symbol]
    except KeyError as exc:
        msg = f"no asset_id for new_symbol={row.new_symbol!r} in symbol map"
        raise ValueError(msg) from exc


def plan_corporate_action_ledger_effects(
    rows: Sequence[CorporateActionRow],
    *,
    portfolio: PortfolioState,
    pit: PitQueryService,
    symbol_to_asset_id: dict[str, str],
    fee_for_notional: Callable[[Decimal], Decimal],
) -> list[CorporateActionLedgerEffect]:
    """
    Return ordered typed ledger effects for the session (shared ``effective_date``).

    ``fee_for_notional`` prices delisting liquidation fees on gross proceeds.
    """

    ordered = sorted(rows, key=_corp_action_sort_key)
    out: list[CorporateActionLedgerEffect] = []

    shares: dict[str, int] = {
        aid: pos.shares for aid, pos in portfolio.positions_by_asset.items()
    }

    for row in ordered:
        if row.action_type == CorporateActionKind.MERGER:
            raise UnsupportedCorporateActionError(
                "merger corporate actions are not supported",
            )
        if row.action_type not in {
            CorporateActionKind.SPLIT,
            CorporateActionKind.CASH_DIVIDEND,
            CorporateActionKind.TICKER_CHANGE,
            CorporateActionKind.DELISTING,
        }:
            raise UnsupportedCorporateActionError(
                f"unsupported corporate action type: {row.action_type!r}",
            )

        if row.action_type == CorporateActionKind.DELISTING:
            q = shares.get(row.asset_id, 0)
            if q > 0:
                px = last_close_before_effective(pit, row.asset_id, row.effective_date)
                gross = Decimal(q) * px
                fees = fee_for_notional(gross)
                out.append(
                    DelistingLedgerEffect(
                        asset_id=row.asset_id,
                        quantity_liquidated=q,
                        cash_proceeds=gross,
                        fees=fees,
                    ),
                )
            shares.pop(row.asset_id, None)
            continue

        if row.action_type == CorporateActionKind.TICKER_CHANGE:
            to_id = resolve_ticker_successor_asset_id(row, symbol_to_asset_id)
            out.append(
                TickerChangeLedgerEffect(
                    from_asset_id=row.asset_id,
                    to_asset_id=to_id,
                ),
            )
            q_old = shares.pop(row.asset_id, 0)
            if q_old:
                shares[to_id] = shares.get(to_id, 0) + q_old
            continue

        if row.action_type == CorporateActionKind.SPLIT:
            ratio = _split_ratio(row)
            out.append(
                SplitLedgerEffect(
                    asset_id=row.asset_id,
                    ratio=ratio,
                ),
            )
            q = shares.get(row.asset_id, 0)
            if q:
                new_q = int(
                    (Decimal(q) * ratio).to_integral_value(rounding=ROUND_DOWN),
                )
                shares[row.asset_id] = new_q
            continue

        if row.dividend_amount is None:
            msg = f"cash_dividend missing dividend_amount for {row.asset_id!r}"
            raise ValueError(msg)
        q = shares.get(row.asset_id, 0)
        credit = Decimal(q) * row.dividend_amount
        if credit > 0:
            out.append(
                DividendLedgerEffect(
                    asset_id=row.asset_id,
                    cash_credited=credit,
                ),
            )

    return out
