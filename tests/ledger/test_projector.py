"""Portfolio projection from ledger events."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from decimal import Decimal

from tradebench.episodes.models import ConvictionEntry, DecisionSnapshot
from tradebench.ledger.events import (
    DayAdvanced,
    DecisionRecorded,
    DelistingLiquidated,
    DividendApplied,
    OrderCancelled,
    OrderFilled,
    OrderSubmitted,
    SplitApplied,
    TickerChanged,
)
from tradebench.ledger.projector import portfolio_state_canonical_json, project
from tradebench.ledger.state import OpenOrder, PortfolioState, Position


def _t(hour: int = 12) -> datetime:
    return datetime(2026, 1, 5, hour, 0, 0, tzinfo=UTC)


def test_replay_produces_identical_canonical_json() -> None:
    events = [
        DividendApplied(
            event_id="e0",
            episode_id="ep",
            event_time=_t(0),
            asset_id="A1",
            cash_credited=Decimal("100000"),
        ),
        OrderSubmitted(
            event_id="e1",
            episode_id="ep",
            event_time=_t(1),
            client_order_id="o1",
            asset_id="A1",
            side="buy",
            quantity=10,
            cash_reserved=Decimal("1000"),
        ),
        OrderFilled(
            event_id="e2",
            episode_id="ep",
            event_time=_t(2),
            client_order_id="o1",
            asset_id="A1",
            side="buy",
            quantity=10,
            avg_fill_price=Decimal("100"),
            fees=Decimal("1"),
        ),
        DayAdvanced(
            event_id="e3",
            episode_id="ep",
            event_time=_t(3),
            session_date=date(2026, 1, 5),
            portfolio_market_value=Decimal("100989"),
            unrealized_pnl=Decimal("89"),
        ),
    ]
    s1 = portfolio_state_canonical_json(project(events))
    s2 = portfolio_state_canonical_json(project(events))
    assert s1 == s2
    assert len(s1) > 0


def test_decision_recorded_does_not_move_portfolio() -> None:
    snap = DecisionSnapshot(
        regime_label="r",
        edge_summary="x",
        intended_exposure=Decimal("0.5"),
        top_convictions=[ConvictionEntry(asset_id="A1", weight=Decimal("0.5"))],
        uncertainty="low",
    )
    with_decision = project(
        [
            DividendApplied(
                event_id="e0",
                episode_id="ep",
                event_time=_t(0),
                asset_id="A1",
                cash_credited=Decimal("100"),
            ),
            DecisionRecorded(
                event_id="e1",
                episode_id="ep",
                event_time=_t(1),
                snapshot=snap,
            ),
        ],
    )
    without = project(
        [
            DividendApplied(
                event_id="e0",
                episode_id="ep",
                event_time=_t(0),
                asset_id="A1",
                cash_credited=Decimal("100"),
            ),
        ],
    )
    assert with_decision == without


def test_buy_fill_updates_cash_and_position() -> None:
    st = project(
        [
            DividendApplied(
                event_id="e0",
                episode_id="ep",
                event_time=_t(0),
                asset_id="A1",
                cash_credited=Decimal("10000"),
            ),
            OrderSubmitted(
                event_id="e1",
                episode_id="ep",
                event_time=_t(1),
                client_order_id="b1",
                asset_id="A1",
                side="buy",
                quantity=50,
                cash_reserved=Decimal("6000"),
            ),
            OrderFilled(
                event_id="e2",
                episode_id="ep",
                event_time=_t(2),
                client_order_id="b1",
                asset_id="A1",
                side="buy",
                quantity=50,
                avg_fill_price=Decimal("100"),
                fees=Decimal("5"),
            ),
        ],
    )
    fill_cost = Decimal("50") * Decimal("100") + Decimal("5")
    assert st.cash == Decimal("10000") - fill_cost
    assert st.positions_by_asset == {
        "A1": Position(shares=50, cost_basis_total=Decimal("5005")),
    }
    assert st.open_orders == {}


def test_sell_fill_realized_pnl() -> None:
    st = project(
        [
            DividendApplied(
                event_id="e0",
                episode_id="ep",
                event_time=_t(0),
                asset_id="A1",
                cash_credited=Decimal("10000"),
            ),
            OrderSubmitted(
                event_id="e1",
                episode_id="ep",
                event_time=_t(1),
                client_order_id="b1",
                asset_id="A1",
                side="buy",
                quantity=100,
                cash_reserved=Decimal("10000"),
            ),
            OrderFilled(
                event_id="e2",
                episode_id="ep",
                event_time=_t(2),
                client_order_id="b1",
                asset_id="A1",
                side="buy",
                quantity=100,
                avg_fill_price=Decimal("100"),
                fees=Decimal("0"),
            ),
            OrderSubmitted(
                event_id="e3",
                episode_id="ep",
                event_time=_t(3),
                client_order_id="s1",
                asset_id="A1",
                side="sell",
                quantity=40,
            ),
            OrderFilled(
                event_id="e4",
                episode_id="ep",
                event_time=_t(4),
                client_order_id="s1",
                asset_id="A1",
                side="sell",
                quantity=40,
                avg_fill_price=Decimal("110"),
                fees=Decimal("2"),
            ),
        ],
    )
    basis_removed = (Decimal("10000") * Decimal("40")) / Decimal("100")
    proceeds = Decimal("40") * Decimal("110") - Decimal("2")
    assert st.realized_pnl == proceeds - basis_removed
    assert st.positions_by_asset["A1"].shares == 60


def test_cancel_refunds_buy_reservation() -> None:
    st = project(
        [
            DividendApplied(
                event_id="e0",
                episode_id="ep",
                event_time=_t(0),
                asset_id="X",
                cash_credited=Decimal("5000"),
            ),
            OrderSubmitted(
                event_id="e1",
                episode_id="ep",
                event_time=_t(1),
                client_order_id="c1",
                asset_id="X",
                side="buy",
                quantity=1,
                cash_reserved=Decimal("3000"),
            ),
            OrderCancelled(
                event_id="e2",
                episode_id="ep",
                event_time=_t(2),
                client_order_id="c1",
            ),
        ],
    )
    assert st.cash == Decimal("5000")
    assert st.open_orders == {}


def test_ticker_change_merges_position_and_open_order() -> None:
    st = project(
        [
            DividendApplied(
                event_id="e0",
                episode_id="ep",
                event_time=_t(0),
                asset_id="OLD",
                cash_credited=Decimal("10000"),
            ),
            OrderSubmitted(
                event_id="e1",
                episode_id="ep",
                event_time=_t(1),
                client_order_id="p",
                asset_id="OLD",
                side="buy",
                quantity=10,
                cash_reserved=Decimal("1000"),
            ),
            TickerChanged(
                event_id="e2",
                episode_id="ep",
                event_time=_t(2),
                from_asset_id="OLD",
                to_asset_id="NEW",
            ),
        ],
    )
    assert "OLD" not in st.positions_by_asset
    assert st.open_orders["p"].asset_id == "NEW"


def test_split_multiplies_shares_preserves_basis() -> None:
    st = project(
        [
            DividendApplied(
                event_id="e0",
                episode_id="ep",
                event_time=_t(0),
                asset_id="S",
                cash_credited=Decimal("10000"),
            ),
            OrderSubmitted(
                event_id="e1",
                episode_id="ep",
                event_time=_t(1),
                client_order_id="b",
                asset_id="S",
                side="buy",
                quantity=10,
                cash_reserved=Decimal("1000"),
            ),
            OrderFilled(
                event_id="e2",
                episode_id="ep",
                event_time=_t(2),
                client_order_id="b",
                asset_id="S",
                side="buy",
                quantity=10,
                avg_fill_price=Decimal("100"),
                fees=Decimal("0"),
            ),
            SplitApplied(
                event_id="e3",
                episode_id="ep",
                event_time=_t(3),
                asset_id="S",
                ratio=Decimal("2"),
            ),
        ],
    )
    assert st.positions_by_asset["S"] == Position(
        shares=20,
        cost_basis_total=Decimal("1000"),
    )


def test_delisting_liquidates_full_position() -> None:
    st = project(
        [
            DividendApplied(
                event_id="e0",
                episode_id="ep",
                event_time=_t(0),
                asset_id="D",
                cash_credited=Decimal("10000"),
            ),
            OrderSubmitted(
                event_id="e1",
                episode_id="ep",
                event_time=_t(1),
                client_order_id="b",
                asset_id="D",
                side="buy",
                quantity=100,
                cash_reserved=Decimal("10000"),
            ),
            OrderFilled(
                event_id="e2",
                episode_id="ep",
                event_time=_t(2),
                client_order_id="b",
                asset_id="D",
                side="buy",
                quantity=100,
                avg_fill_price=Decimal("50"),
                fees=Decimal("0"),
            ),
            DelistingLiquidated(
                event_id="e3",
                episode_id="ep",
                event_time=_t(3),
                asset_id="D",
                quantity_liquidated=100,
                cash_proceeds=Decimal("4200"),
                fees=Decimal("0"),
            ),
        ],
    )
    assert "D" not in st.positions_by_asset
    # Cash after buy fill is 5000; delisting adds 4200 net proceeds.
    assert st.cash == Decimal("9200")
    assert st.realized_pnl == Decimal("4200") - Decimal("5000")


def test_portfolio_state_json_sorts_mapping_keys() -> None:
    """Serialized dict key order must not depend on mutation order."""

    st = PortfolioState(
        cash=Decimal("1"),
        positions_by_asset={
            "Z": Position(shares=1, cost_basis_total=Decimal("2")),
            "A": Position(shares=3, cost_basis_total=Decimal("4")),
        },
        open_orders={
            "b": OpenOrder(
                client_order_id="b",
                asset_id="A",
                side="buy",
                quantity=1,
                cash_reserved=Decimal("0"),
            ),
            "a": OpenOrder(
                client_order_id="a",
                asset_id="Z",
                side="sell",
                quantity=2,
            ),
        },
        last_valuation=Decimal("9"),
        realized_pnl=Decimal("0"),
        unrealized_pnl=Decimal("0"),
        last_session_date=date(2026, 1, 1),
    )
    raw = portfolio_state_canonical_json(st)
    parsed = json.loads(raw)
    assert list(parsed["positions_by_asset"]) == ["A", "Z"]
    assert list(parsed["open_orders"]) == ["a", "b"]
