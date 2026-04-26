"""Canonical row model validation (PIT timestamps and visibility bounds)."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta, timezone
from decimal import Decimal

import pytest
from pydantic import ValidationError

from tradebench.data.models import (
    AssetMasterRow,
    CorporateActionKind,
    CorporateActionRow,
    DailyBarRow,
    FundamentalRow,
)


def _day(d: date, *, hour: int = 12) -> datetime:
    return datetime(d.year, d.month, d.day, hour, 0, 0, tzinfo=UTC)


def test_daily_bar_requires_available_at() -> None:
    with pytest.raises(ValidationError) as exc:
        DailyBarRow(
            asset_id="A",
            session_date=date(2020, 1, 2),
            open=Decimal("1"),
            high=Decimal("2"),
            low=Decimal("1"),
            close=Decimal("2"),
            volume=100,
            dollar_volume=Decimal("200"),
        )
    assert "available_at" in str(exc.value).lower()


def test_daily_bar_rejects_available_at_after_session_boundary() -> None:
    d = date(2020, 1, 2)
    bad = _day(d) + timedelta(days=1)
    with pytest.raises(ValidationError, match="available_at exceeds"):
        DailyBarRow(
            asset_id="A",
            session_date=d,
            open=Decimal("1"),
            high=Decimal("2"),
            low=Decimal("1"),
            close=Decimal("2"),
            volume=100,
            dollar_volume=Decimal("200"),
            available_at=bad,
        )


def test_daily_bar_treats_naive_available_at_as_utc() -> None:
    row = DailyBarRow(
        asset_id="A",
        session_date=date(2020, 1, 2),
        open=Decimal("1"),
        high=Decimal("2"),
        low=Decimal("1"),
        close=Decimal("2"),
        volume=100,
        dollar_volume=Decimal("200"),
        available_at=datetime(2020, 1, 2, 12, 0, 0),
    )
    assert row.available_at == datetime(2020, 1, 2, 12, 0, 0, tzinfo=UTC)


def test_daily_bar_normalizes_aware_available_at_to_utc() -> None:
    eastern = timezone(timedelta(hours=-5))
    row = DailyBarRow(
        asset_id="A",
        session_date=date(2020, 1, 2),
        open=Decimal("1"),
        high=Decimal("2"),
        low=Decimal("1"),
        close=Decimal("2"),
        volume=100,
        dollar_volume=Decimal("200"),
        available_at=datetime(2020, 1, 2, 15, 0, 0, tzinfo=eastern),
    )
    assert row.available_at == datetime(2020, 1, 2, 20, 0, 0, tzinfo=UTC)


def test_fundamental_requires_available_at() -> None:
    with pytest.raises(ValidationError):
        FundamentalRow(
            asset_id="A",
            fiscal_period_end=date(2019, 12, 31),
        )


def test_fundamental_rejects_available_at_before_period_end() -> None:
    end = date(2019, 12, 31)
    with pytest.raises(
        ValidationError,
        match="available_at precedes fiscal_period_end",
    ):
        FundamentalRow(
            asset_id="A",
            fiscal_period_end=end,
            available_at=_day(date(2019, 11, 1)),
        )


def test_asset_master_requires_available_at() -> None:
    with pytest.raises(ValidationError):
        AssetMasterRow(
            asset_id="A",
            symbol="A",
            snapshot_date=date(2020, 1, 2),
        )


def test_asset_master_rejects_available_at_after_snapshot_boundary() -> None:
    snap = date(2020, 1, 2)
    with pytest.raises(ValidationError, match="available_at exceeds"):
        AssetMasterRow(
            asset_id="A",
            symbol="A",
            snapshot_date=snap,
            available_at=_day(snap) + timedelta(days=1),
        )


def test_corporate_action_requires_available_at() -> None:
    with pytest.raises(ValidationError):
        CorporateActionRow(
            asset_id="A",
            action_type=CorporateActionKind.CASH_DIVIDEND,
            effective_date=date(2020, 3, 1),
        )


def test_corporate_action_rejects_available_at_after_effective_boundary() -> None:
    eff = date(2020, 3, 1)
    with pytest.raises(ValidationError, match="available_at exceeds"):
        CorporateActionRow(
            asset_id="A",
            action_type=CorporateActionKind.SPLIT,
            effective_date=eff,
            split_from=Decimal("1"),
            split_to=Decimal("2"),
            available_at=_day(eff) + timedelta(days=2),
        )


def test_corporate_action_metadata_field_is_documented_as_vendor_specific() -> None:
    description = CorporateActionRow.model_fields["metadata"].description
    assert description is not None
    assert "vendor" in description.lower()
