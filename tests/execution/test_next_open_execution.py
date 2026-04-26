"""Next-open execution, explicit costs, and accounting reconciliation."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from tradebench.data.query import PitQueryService
from tradebench.data.sample_dataset import ASSET_LIQUID, SAMPLE_DATASET_VERSION
from tradebench.episodes.models import EpisodeManifest
from tradebench.execution.costs import (
    CostModel,
    compute_fee_total,
    compute_slippage_bps,
    fill_price_from_open,
)
from tradebench.execution.engine import (
    OrderIntent,
    advance_trading_session,
    portfolio_market_value,
)
from tradebench.ledger.events import (
    DayAdvanced,
    DividendApplied,
    LedgerEvent,
    OrderFilled,
)
from tradebench.ledger.projector import project


def _manifest(sample_dataset_root: Path) -> EpisodeManifest:
    path = (
        sample_dataset_root
        / "catalog"
        / SAMPLE_DATASET_VERSION
        / "episode_manifests"
        / "sample_ep_001.json"
    )
    return EpisodeManifest.model_validate_json(path.read_text(encoding="utf-8"))


def _last_close(pit: PitQueryService, asset_id: str, session_date: date) -> Decimal:
    df = pit.get_bars([asset_id], session_date, lookback_days=1)
    row = df.iloc[-1]
    return Decimal(str(row["close"]))


def test_fills_use_next_session_open_not_decision_close(
    pit_query_service: PitQueryService,
    sample_dataset_root: Path,
) -> None:
    """Buy decided on ``t`` must fill at ``t+1`` open (adjusted), not at ``t`` close."""

    pit = pit_query_service
    manifest = _manifest(sample_dataset_root)
    dec = date(2020, 1, 2)
    exe = date(2020, 1, 3)
    model = CostModel(
        commission_floor=Decimal("1"),
        fee_bps=Decimal("10"),
        slippage_bps_cap=Decimal("50"),
    )
    prior: list[LedgerEvent] = [
        DividendApplied(
            event_id="boot",
            episode_id="ep",
            event_time=datetime(2020, 1, 1, 12, 0, 0, tzinfo=UTC),
            asset_id=ASSET_LIQUID,
            cash_credited=Decimal("100000"),
        ),
    ]
    step = advance_trading_session(
        pit=pit,
        manifest=manifest,
        episode_id="ep",
        decision_date=dec,
        execution_date=exe,
        prior_events=prior,
        new_orders=[
            OrderIntent(
                client_order_id="b1",
                asset_id=ASSET_LIQUID,
                side="buy",
                quantity=10,
            ),
        ],
        cost_model=model,
        base_event_time=datetime(2020, 1, 2, 16, 0, 0, tzinfo=UTC),
    )
    fills = [e for e in step if isinstance(e, OrderFilled)]
    assert len(fills) == 1
    fill = fills[0]

    dec_close = _last_close(pit, ASSET_LIQUID, dec)
    exe_open = Decimal(str(pit.get_bars([ASSET_LIQUID], exe, 1).iloc[-1]["open"]))
    assert dec_close == Decimal("100")
    assert exe_open == Decimal("99.75")

    bar_dec = pit.get_bars([ASSET_LIQUID], dec, 1).iloc[-1]
    dv = Decimal(str(bar_dec["dollar_volume"]))
    slip = compute_slippage_bps(
        model,
        order_notional=Decimal(fill.quantity) * exe_open,
        trailing_dollar_volume=dv,
    )
    expected_px = fill_price_from_open(exe_open, "buy", slip)
    assert fill.avg_fill_price == expected_px
    assert fill.avg_fill_price != fill_price_from_open(dec_close, "buy", slip)


def test_cash_debits_include_commission_and_bps_fees(
    pit_query_service: PitQueryService,
    sample_dataset_root: Path,
) -> None:
    pit = pit_query_service
    manifest = _manifest(sample_dataset_root)
    model = CostModel(
        commission_floor=Decimal("2"),
        fee_bps=Decimal("25"),
        slippage_bps_cap=Decimal("100"),
    )
    prior: list[LedgerEvent] = [
        DividendApplied(
            event_id="boot",
            episode_id="ep",
            event_time=datetime(2020, 1, 1, 12, 0, 0, tzinfo=UTC),
            asset_id=ASSET_LIQUID,
            cash_credited=Decimal("100000"),
        ),
    ]
    step = advance_trading_session(
        pit=pit,
        manifest=manifest,
        episode_id="ep",
        decision_date=date(2020, 1, 2),
        execution_date=date(2020, 1, 3),
        prior_events=prior,
        new_orders=[
            OrderIntent(
                client_order_id="b1",
                asset_id=ASSET_LIQUID,
                side="buy",
                quantity=5,
            ),
        ],
        cost_model=model,
        base_event_time=datetime(2020, 1, 2, 16, 0, 0, tzinfo=UTC),
    )
    fill = next(e for e in step if isinstance(e, OrderFilled))
    fill_notional = fill.avg_fill_price * Decimal(fill.quantity)
    expected_fees = compute_fee_total(model, fill_notional=fill_notional)
    assert fill.fees == expected_fees

    st = project([*prior, *step])
    assert st.cash >= Decimal("0")


def test_scripted_episode_reconciles_after_each_day_advanced(
    pit_query_service: PitQueryService,
    sample_dataset_root: Path,
) -> None:
    """``DayAdvanced.portfolio_market_value`` matches cash + marks from projection."""

    pit = pit_query_service
    manifest = _manifest(sample_dataset_root)
    model = CostModel(
        commission_floor=Decimal("1"),
        fee_bps=Decimal("10"),
        slippage_bps_cap=Decimal("30"),
    )
    events: list[LedgerEvent] = [
        DividendApplied(
            event_id="boot",
            episode_id="ep",
            event_time=datetime(2020, 1, 1, 12, 0, 0, tzinfo=UTC),
            asset_id=ASSET_LIQUID,
            cash_credited=Decimal("100000"),
        ),
    ]
    days = [
        (date(2020, 1, 2), date(2020, 1, 3)),
        (date(2020, 1, 3), date(2020, 1, 6)),
    ]
    for dec, exe in days:
        step = advance_trading_session(
            pit=pit,
            manifest=manifest,
            episode_id="ep",
            decision_date=dec,
            execution_date=exe,
            prior_events=events,
            new_orders=[
                OrderIntent(
                    client_order_id=f"buy-{exe}",
                    asset_id=ASSET_LIQUID,
                    side="buy",
                    quantity=1,
                ),
            ],
            cost_model=model,
            base_event_time=datetime.combine(dec, datetime.min.time()).replace(
                tzinfo=UTC,
            )
            + timedelta(hours=16),
        )
        events.extend(step)
        adv = next(e for e in step if isinstance(e, DayAdvanced))
        st = project(events)
        closes = {
            ASSET_LIQUID: _last_close(pit, ASSET_LIQUID, exe),
        }
        recomputed = portfolio_market_value(st, closes)
        assert adv.portfolio_market_value == recomputed
