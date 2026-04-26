"""Pure portfolio projection from append-only ledger events."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import date
from decimal import ROUND_DOWN, Decimal
from typing import Any

from tradebench.ledger.events import (
    DayAdvanced,
    DecisionRecorded,
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
from tradebench.ledger.state import OpenOrder, PortfolioState, Position


def _sort_json_mapping(obj: Any) -> Any:
    """Recursively sort mapping keys so nested objects compare byte-for-byte."""

    if isinstance(obj, dict):
        return {k: _sort_json_mapping(obj[k]) for k in sorted(obj)}
    if isinstance(obj, list):
        return [_sort_json_mapping(v) for v in obj]
    return obj


def portfolio_state_canonical_json(state: PortfolioState) -> str:
    """Deterministic JSON for byte-stable regression tests."""

    payload: dict[str, Any] = _sort_json_mapping(state.model_dump(mode="json"))
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _allocate_cost_basis(pos: Position, quantity: int) -> tuple[Decimal, Position]:
    """Remove quantity shares and return allocated cost basis and updated position."""

    if quantity < 0 or quantity > pos.shares:
        msg = "liquidation quantity exceeds position"
        raise ValueError(msg)
    if quantity == 0:
        return Decimal("0"), pos
    if pos.shares == 0:
        return Decimal("0"), pos
    allocated = (pos.cost_basis_total * quantity) / pos.shares
    new_shares = pos.shares - quantity
    new_cost = pos.cost_basis_total - allocated
    return allocated, Position(shares=new_shares, cost_basis_total=new_cost)


class _MutablePortfolio:
    """Internal accumulator; not exposed outside projection."""

    __slots__ = (
        "cash",
        "last_session_date",
        "last_valuation",
        "open_orders",
        "positions",
        "realized_pnl",
        "seen_submit_keys",
        "unrealized_pnl",
    )

    def __init__(self) -> None:
        self.cash = Decimal("0")
        self.positions: dict[str, Position] = {}
        self.open_orders: dict[str, OpenOrder] = {}
        self.last_valuation: Decimal | None = None
        self.realized_pnl = Decimal("0")
        self.unrealized_pnl = Decimal("0")
        self.last_session_date: date | None = None
        self.seen_submit_keys: set[tuple[str, str]] = set()

    def to_frozen(self) -> PortfolioState:
        return PortfolioState(
            cash=self.cash,
            positions_by_asset={k: self.positions[k] for k in sorted(self.positions)},
            open_orders={k: self.open_orders[k] for k in sorted(self.open_orders)},
            last_valuation=self.last_valuation,
            realized_pnl=self.realized_pnl,
            unrealized_pnl=self.unrealized_pnl,
            last_session_date=self.last_session_date,
        )

    def submit_key(self, episode_id: str, client_order_id: str) -> tuple[str, str]:
        """Append-only idempotency key for order submission admission."""

        return (episode_id, client_order_id)

    def apply_decision_recorded(self, _event: DecisionRecorded) -> None:
        return

    def apply_order_submitted(self, event: OrderSubmitted) -> None:
        key = self.submit_key(event.episode_id, event.client_order_id)
        if key in self.seen_submit_keys:
            return
        self.seen_submit_keys.add(key)
        if event.side == "buy":
            self.cash -= event.cash_reserved
        self.open_orders[event.client_order_id] = OpenOrder(
            client_order_id=event.client_order_id,
            asset_id=event.asset_id,
            side=event.side,
            quantity=event.quantity,
            cash_reserved=event.cash_reserved if event.side == "buy" else Decimal("0"),
        )

    def _release_open_order(self, client_order_id: str) -> OpenOrder | None:
        return self.open_orders.pop(client_order_id, None)

    def _close_open_order(self, client_order_id: str) -> None:
        """Remove an open order and refund any reserved buy cash."""

        open_order = self._release_open_order(client_order_id)
        if open_order is None:
            return
        if open_order.side == "buy":
            self.cash += open_order.cash_reserved

    def apply_order_rejected(self, event: OrderRejected) -> None:
        self._close_open_order(event.client_order_id)

    def apply_order_cancelled(self, event: OrderCancelled) -> None:
        self._close_open_order(event.client_order_id)

    def apply_order_filled(self, event: OrderFilled) -> None:
        open_order = self.open_orders.get(event.client_order_id)
        if open_order is None:
            msg = f"OrderFilled for unknown client_order_id={event.client_order_id!r}"
            raise ValueError(msg)
        if open_order.side != event.side or open_order.asset_id != event.asset_id:
            msg = "OrderFilled does not match open order side/asset"
            raise ValueError(msg)
        if open_order.quantity != event.quantity:
            msg = "OrderFilled quantity must match open order"
            raise ValueError(msg)

        del self.open_orders[event.client_order_id]

        if event.side == "buy":
            self.cash += open_order.cash_reserved
            total_cost = event.quantity * event.avg_fill_price + event.fees
            self.cash -= total_cost
            empty = Position(shares=0, cost_basis_total=Decimal("0"))
            pos = self.positions.get(event.asset_id, empty)
            new_shares = pos.shares + event.quantity
            new_basis = pos.cost_basis_total + total_cost
            self.positions[event.asset_id] = Position(
                shares=new_shares,
                cost_basis_total=new_basis,
            )
        else:
            sell_pos = self.positions.get(event.asset_id)
            if sell_pos is None or sell_pos.shares < event.quantity:
                msg = "Sell fill exceeds available position"
                raise ValueError(msg)
            allocated_basis, new_pos = _allocate_cost_basis(sell_pos, event.quantity)
            proceeds = event.quantity * event.avg_fill_price - event.fees
            self.cash += proceeds
            self.realized_pnl += proceeds - allocated_basis
            if new_pos.shares == 0:
                self.positions.pop(event.asset_id, None)
            else:
                self.positions[event.asset_id] = new_pos

    def apply_dividend_applied(self, event: DividendApplied) -> None:
        self.cash += event.cash_credited

    def apply_split_applied(self, event: SplitApplied) -> None:
        pos = self.positions.get(event.asset_id)
        if pos is None or pos.shares == 0:
            return
        new_shares = int(
            (Decimal(pos.shares) * event.ratio).to_integral_value(rounding=ROUND_DOWN),
        )
        self.positions[event.asset_id] = Position(
            shares=new_shares,
            cost_basis_total=pos.cost_basis_total,
        )

    def apply_ticker_changed(self, event: TickerChanged) -> None:
        for oid, oo in list(self.open_orders.items()):
            if oo.asset_id == event.from_asset_id:
                self.open_orders[oid] = OpenOrder(
                    client_order_id=oo.client_order_id,
                    asset_id=event.to_asset_id,
                    side=oo.side,
                    quantity=oo.quantity,
                    cash_reserved=oo.cash_reserved,
                )

        pos = self.positions.pop(event.from_asset_id, None)
        if pos is None:
            return
        existing = self.positions.get(event.to_asset_id)
        if existing is None:
            self.positions[event.to_asset_id] = pos
            return
        merged = Position(
            shares=existing.shares + pos.shares,
            cost_basis_total=existing.cost_basis_total + pos.cost_basis_total,
        )
        self.positions[event.to_asset_id] = merged

    def apply_delisting_liquidated(self, event: DelistingLiquidated) -> None:
        pos = self.positions.get(event.asset_id)
        if pos is None or pos.shares == 0:
            return
        if event.quantity_liquidated != pos.shares:
            msg = "DelistingLiquidated quantity must match full position"
            raise ValueError(msg)
        allocated_basis, _ = _allocate_cost_basis(pos, event.quantity_liquidated)
        net = event.cash_proceeds - event.fees
        self.cash += net
        self.realized_pnl += net - allocated_basis
        self.positions.pop(event.asset_id, None)

    def apply_day_advanced(self, event: DayAdvanced) -> None:
        self.last_session_date = event.session_date
        if event.portfolio_market_value is not None:
            self.last_valuation = event.portfolio_market_value
        if event.unrealized_pnl is not None:
            self.unrealized_pnl = event.unrealized_pnl


def project(events: Sequence[LedgerEvent]) -> PortfolioState:
    """
    Fold ledger events into a portfolio snapshot.

    Pure function: no I/O, no shared mutable globals. Deterministic for the same
    event sequence.
    """

    acc = _MutablePortfolio()
    for event in events:
        if isinstance(event, DecisionRecorded):
            acc.apply_decision_recorded(event)
        elif isinstance(event, OrderSubmitted):
            acc.apply_order_submitted(event)
        elif isinstance(event, OrderRejected):
            acc.apply_order_rejected(event)
        elif isinstance(event, OrderCancelled):
            acc.apply_order_cancelled(event)
        elif isinstance(event, OrderFilled):
            acc.apply_order_filled(event)
        elif isinstance(event, DividendApplied):
            acc.apply_dividend_applied(event)
        elif isinstance(event, SplitApplied):
            acc.apply_split_applied(event)
        elif isinstance(event, TickerChanged):
            acc.apply_ticker_changed(event)
        elif isinstance(event, DelistingLiquidated):
            acc.apply_delisting_liquidated(event)
        elif isinstance(event, DayAdvanced):
            acc.apply_day_advanced(event)
        else:
            msg = f"unknown ledger event type: {type(event).__name__}"
            raise TypeError(msg)
    return acc.to_frozen()
