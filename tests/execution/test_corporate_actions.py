"""Corporate actions: splits, dividends, ticker, delisting, unsupported kinds."""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from tradebench.data.models import (
    CorporateActionKind,
    CorporateActionRow,
    utc_end_of_day,
)
from tradebench.data.query import PitQueryService
from tradebench.data.sample_dataset import (
    ASSET_DEAD,
    ASSET_LIQUID,
    ASSET_RENAME_NEW,
    ASSET_RENAME_OLD,
    SAMPLE_DATASET_VERSION,
    SYMBOL_RENAME_NEW,
)
from tradebench.episodes.models import EpisodeManifest
from tradebench.execution.corporate_actions import (
    TickerChangeLedgerEffect,
    UnsupportedCorporateActionError,
    plan_corporate_action_ledger_effects,
)
from tradebench.execution.costs import CostModel
from tradebench.execution.engine import OrderIntent, advance_trading_session
from tradebench.ledger.events import (
    DividendApplied,
    LedgerEvent,
    OrderFilled,
    OrderSubmitted,
    TickerChanged,
)
from tradebench.ledger.projector import project
from tradebench.ledger.state import PortfolioState


def _manifest(sample_dataset_root: Path) -> EpisodeManifest:
    path = (
        sample_dataset_root
        / "catalog"
        / SAMPLE_DATASET_VERSION
        / "episode_manifests"
        / "sample_ep_001.json"
    )
    return EpisodeManifest.model_validate_json(path.read_text(encoding="utf-8"))


def test_split_doubles_shares_and_preserves_cost_basis(
    pit_query_service: PitQueryService,
    sample_dataset_root: Path,
) -> None:
    model = CostModel(
        commission_floor=Decimal("0"),
        fee_bps=Decimal("0"),
        slippage_bps_cap=Decimal("0"),
    )
    pit = pit_query_service
    manifest = _manifest(sample_dataset_root)
    prior: list[LedgerEvent] = [
        DividendApplied(
            event_id="boot",
            episode_id="ep",
            event_time=datetime(2020, 1, 1, 12, 0, 0, tzinfo=UTC),
            asset_id=ASSET_LIQUID,
            cash_credited=Decimal("100000"),
        ),
    ]
    prior.extend(
        advance_trading_session(
            pit=pit,
            manifest=manifest,
            episode_id="ep",
            decision_date=date(2020, 1, 2),
            execution_date=date(2020, 1, 3),
            prior_events=prior,
            new_orders=[
                OrderIntent(
                    client_order_id="b0",
                    asset_id=ASSET_LIQUID,
                    side="buy",
                    quantity=10,
                ),
            ],
            cost_model=model,
            base_event_time=datetime(2020, 1, 2, 16, 0, 0, tzinfo=UTC),
        ),
    )
    before = project(prior)
    basis_before = before.positions_by_asset[ASSET_LIQUID].cost_basis_total

    split_step = advance_trading_session(
        pit=pit,
        manifest=manifest,
        episode_id="ep",
        decision_date=date(2020, 1, 31),
        execution_date=date(2020, 2, 3),
        prior_events=prior,
        new_orders=[],
        cost_model=model,
        base_event_time=datetime(2020, 1, 31, 16, 0, 0, tzinfo=UTC),
    )
    after = project([*prior, *split_step])
    pos = after.positions_by_asset[ASSET_LIQUID]
    assert pos.shares == 20
    assert pos.cost_basis_total == basis_before


def test_dividend_credits_cash_on_effective_session(
    pit_query_service: PitQueryService,
    sample_dataset_root: Path,
) -> None:
    model = CostModel(
        commission_floor=Decimal("0"),
        fee_bps=Decimal("0"),
        slippage_bps_cap=Decimal("0"),
    )
    pit = pit_query_service
    manifest = _manifest(sample_dataset_root)
    prior: list[LedgerEvent] = [
        DividendApplied(
            event_id="boot",
            episode_id="ep",
            event_time=datetime(2020, 1, 1, 12, 0, 0, tzinfo=UTC),
            asset_id=ASSET_LIQUID,
            cash_credited=Decimal("100000"),
        ),
        OrderSubmitted(
            event_id="os",
            episode_id="ep",
            event_time=datetime(2020, 1, 2, 12, 0, 0, tzinfo=UTC),
            client_order_id="manual",
            asset_id=ASSET_LIQUID,
            side="buy",
            quantity=100,
            cash_reserved=Decimal("100000"),
        ),
        OrderFilled(
            event_id="of",
            episode_id="ep",
            event_time=datetime(2020, 1, 2, 12, 1, 0, tzinfo=UTC),
            client_order_id="manual",
            asset_id=ASSET_LIQUID,
            side="buy",
            quantity=100,
            avg_fill_price=Decimal("100"),
            fees=Decimal("0"),
        ),
    ]
    cash_before = project(prior).cash
    step = advance_trading_session(
        pit=pit,
        manifest=manifest,
        episode_id="ep",
        decision_date=date(2020, 2, 28),
        execution_date=date(2020, 3, 2),
        prior_events=prior,
        new_orders=[],
        cost_model=model,
        base_event_time=datetime(2020, 2, 28, 16, 0, 0, tzinfo=UTC),
    )
    after = project([*prior, *step])
    assert after.cash == cash_before + Decimal("42")


def test_delisting_liquidates_at_last_close_before_effective_date(
    pit_query_service: PitQueryService,
    sample_dataset_root: Path,
) -> None:
    """
    v0 policy: full exit at last available daily close with ``session_date`` strictly
    before the delisting ``effective_date``, with the same fee stack as regular sells.
    """

    model = CostModel(
        commission_floor=Decimal("1"),
        fee_bps=Decimal("10"),
        slippage_bps_cap=Decimal("0"),
    )
    pit = pit_query_service
    manifest = _manifest(sample_dataset_root)
    prior: list[LedgerEvent] = [
        DividendApplied(
            event_id="boot",
            episode_id="ep",
            event_time=datetime(2020, 1, 1, 12, 0, 0, tzinfo=UTC),
            asset_id=ASSET_DEAD,
            cash_credited=Decimal("50000"),
        ),
        OrderSubmitted(
            event_id="os",
            episode_id="ep",
            event_time=datetime(2020, 3, 9, 12, 0, 0, tzinfo=UTC),
            client_order_id="dead",
            asset_id=ASSET_DEAD,
            side="buy",
            quantity=200,
            cash_reserved=Decimal("50000"),
        ),
        OrderFilled(
            event_id="of",
            episode_id="ep",
            event_time=datetime(2020, 3, 9, 12, 1, 0, tzinfo=UTC),
            client_order_id="dead",
            asset_id=ASSET_DEAD,
            side="buy",
            quantity=200,
            avg_fill_price=Decimal("5"),
            fees=Decimal("0"),
        ),
    ]
    step = advance_trading_session(
        pit=pit,
        manifest=manifest,
        episode_id="ep",
        decision_date=date(2020, 3, 13),
        execution_date=date(2020, 3, 16),
        prior_events=prior,
        new_orders=[],
        cost_model=model,
        base_event_time=datetime(2020, 3, 13, 16, 0, 0, tzinfo=UTC),
    )
    final = project([*prior, *step])
    assert ASSET_DEAD not in final.positions_by_asset
    last_close = Decimal("5")
    gross = Decimal("200") * last_close
    expected_fee = model.commission_floor + gross * model.fee_bps / Decimal("10000")
    assert final.cash == project(prior).cash + gross - expected_fee


def test_ticker_change_moves_position_via_planner_and_projector(
    pit_query_service: PitQueryService,
) -> None:
    pit = pit_query_service
    prior_events: list[LedgerEvent] = [
        DividendApplied(
            event_id="boot",
            episode_id="ep",
            event_time=datetime(2020, 1, 1, 12, 0, 0, tzinfo=UTC),
            asset_id=ASSET_RENAME_OLD,
            cash_credited=Decimal("10000"),
        ),
        OrderSubmitted(
            event_id="os",
            episode_id="ep",
            event_time=datetime(2020, 1, 2, 12, 0, 0, tzinfo=UTC),
            client_order_id="o_old",
            asset_id=ASSET_RENAME_OLD,
            side="buy",
            quantity=40,
            cash_reserved=Decimal("10000"),
        ),
        OrderFilled(
            event_id="of",
            episode_id="ep",
            event_time=datetime(2020, 1, 2, 12, 1, 0, tzinfo=UTC),
            client_order_id="o_old",
            asset_id=ASSET_RENAME_OLD,
            side="buy",
            quantity=40,
            avg_fill_price=Decimal("12"),
            fees=Decimal("0"),
        ),
    ]
    st = project(prior_events)
    row = CorporateActionRow(
        asset_id=ASSET_RENAME_OLD,
        action_type=CorporateActionKind.TICKER_CHANGE,
        effective_date=date(2020, 2, 3),
        ex_date=None,
        new_symbol=SYMBOL_RENAME_NEW,
        metadata={},
        available_at=utc_end_of_day(date(2020, 2, 3)),
    )
    planned = plan_corporate_action_ledger_effects(
        [row],
        portfolio=st,
        pit=pit,
        symbol_to_asset_id={SYMBOL_RENAME_NEW: ASSET_RENAME_NEW},
        fee_for_notional=lambda _n: Decimal("0"),
    )
    assert planned == [
        TickerChangeLedgerEffect(
            from_asset_id=ASSET_RENAME_OLD,
            to_asset_id=ASSET_RENAME_NEW,
        ),
    ]
    st2 = project(
        [
            *prior_events,
            TickerChanged(
                event_id="t1",
                episode_id="ep",
                event_time=datetime(2020, 2, 3, 12, 0, 0, tzinfo=UTC),
                from_asset_id=ASSET_RENAME_OLD,
                to_asset_id=ASSET_RENAME_NEW,
            ),
        ],
    )
    assert ASSET_RENAME_OLD not in st2.positions_by_asset
    assert st2.positions_by_asset[ASSET_RENAME_NEW].shares == 40


def test_merger_action_fails_closed() -> None:
    row = CorporateActionRow(
        asset_id="tb_merger",
        action_type=CorporateActionKind.MERGER,
        effective_date=date(2020, 1, 2),
        available_at=utc_end_of_day(date(2020, 1, 2)),
    )
    with pytest.raises(UnsupportedCorporateActionError, match="merger"):
        plan_corporate_action_ledger_effects(
            [row],
            portfolio=PortfolioState.initial(),
            pit=None,  # type: ignore[arg-type]
            symbol_to_asset_id={},
            fee_for_notional=lambda _n: Decimal("0"),
        )
