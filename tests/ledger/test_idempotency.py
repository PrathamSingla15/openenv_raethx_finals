"""Idempotent order admission and replay convergence."""

from __future__ import annotations

from copy import deepcopy
from datetime import UTC, date, datetime
from decimal import Decimal

from tradebench.ledger.events import (
    DayAdvanced,
    DividendApplied,
    OrderCancelled,
    OrderFilled,
    OrderSubmitted,
)
from tradebench.ledger.projector import portfolio_state_canonical_json, project


def _t(m: int = 0) -> datetime:
    return datetime(2026, 2, 1, 15, m, 0, tzinfo=UTC)


def test_duplicate_order_submitted_does_not_double_spend_cash() -> None:
    single_submit = [
        DividendApplied(
            event_id="d0",
            episode_id="ep",
            event_time=_t(0),
            asset_id="A",
            cash_credited=Decimal("10000"),
        ),
        OrderSubmitted(
            event_id="s1",
            episode_id="ep",
            event_time=_t(1),
            client_order_id="dup",
            asset_id="A",
            side="buy",
            quantity=10,
            cash_reserved=Decimal("5000"),
        ),
        OrderFilled(
            event_id="f1",
            episode_id="ep",
            event_time=_t(2),
            client_order_id="dup",
            asset_id="A",
            side="buy",
            quantity=10,
            avg_fill_price=Decimal("100"),
            fees=Decimal("0"),
        ),
        DayAdvanced(
            event_id="adv",
            episode_id="ep",
            event_time=_t(3),
            session_date=date(2026, 2, 1),
            portfolio_market_value=Decimal("5000"),
        ),
    ]
    duplicate_submit = [
        single_submit[0],
        single_submit[1],
        OrderSubmitted(
            event_id="s1b",
            episode_id="ep",
            event_time=_t(4),
            client_order_id="dup",
            asset_id="A",
            side="buy",
            quantity=10,
            cash_reserved=Decimal("5000"),
        ),
        single_submit[2],
        single_submit[3],
    ]
    a = portfolio_state_canonical_json(project(single_submit))
    b = portfolio_state_canonical_json(project(duplicate_submit))
    assert a == b


def test_cancelled_order_id_cannot_be_reused_in_same_episode() -> None:
    """
    `client_order_id` is single-use for append-only replay, even after cancel.
    """

    events = [
        DividendApplied(
            event_id="d0",
            episode_id="ep",
            event_time=_t(20),
            asset_id="A",
            cash_credited=Decimal("10000"),
        ),
        OrderSubmitted(
            event_id="s1",
            episode_id="ep",
            event_time=_t(21),
            client_order_id="reuse",
            asset_id="A",
            side="buy",
            quantity=10,
            cash_reserved=Decimal("3000"),
        ),
        OrderCancelled(
            event_id="c1",
            episode_id="ep",
            event_time=_t(22),
            client_order_id="reuse",
        ),
        OrderSubmitted(
            event_id="s2",
            episode_id="ep",
            event_time=_t(23),
            client_order_id="reuse",
            asset_id="A",
            side="buy",
            quantity=10,
            cash_reserved=Decimal("3000"),
        ),
    ]

    state = project(events)

    assert state.cash == Decimal("10000")
    assert state.open_orders == {}


def test_same_events_replayed_on_fresh_projector_match() -> None:
    """Two fresh projections of the same events match (deterministic replay)."""

    base = [
        DividendApplied(
            event_id="d0",
            episode_id="ep",
            event_time=_t(10),
            asset_id="A",
            cash_credited=Decimal("50000"),
        ),
        OrderSubmitted(
            event_id="o1",
            episode_id="ep",
            event_time=_t(11),
            client_order_id="c1",
            asset_id="A",
            side="buy",
            quantity=100,
            cash_reserved=Decimal("20000"),
        ),
        OrderFilled(
            event_id="f1",
            episode_id="ep",
            event_time=_t(12),
            client_order_id="c1",
            asset_id="A",
            side="buy",
            quantity=100,
            avg_fill_price=Decimal("200"),
            fees=Decimal("10"),
        ),
        OrderSubmitted(
            event_id="o2",
            episode_id="ep",
            event_time=_t(13),
            client_order_id="c2",
            asset_id="A",
            side="sell",
            quantity=30,
        ),
        OrderFilled(
            event_id="f2",
            episode_id="ep",
            event_time=_t(14),
            client_order_id="c2",
            asset_id="A",
            side="sell",
            quantity=30,
            avg_fill_price=Decimal("210"),
            fees=Decimal("1"),
        ),
        DayAdvanced(
            event_id="adv",
            episode_id="ep",
            event_time=_t(15),
            session_date=date(2026, 2, 2),
            portfolio_market_value=Decimal("45000"),
            unrealized_pnl=Decimal("100"),
        ),
    ]
    first = portfolio_state_canonical_json(project(base))
    second = portfolio_state_canonical_json(project(deepcopy(base)))
    assert first == second
