"""Canonical point-in-time row models for TradeBench datasets."""

from __future__ import annotations

from datetime import UTC, date, datetime, time
from decimal import Decimal
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def utc_end_of_day(d: date) -> datetime:
    """UTC end-of-session bound for same-day PIT visibility checks."""

    return datetime.combine(d, time(23, 59, 59, 999999), tzinfo=UTC)


def utc_start_of_day(d: date) -> datetime:
    return datetime.combine(d, time.min, tzinfo=UTC)


def ensure_utc(dt: datetime) -> datetime:
    """Normalize timestamps to UTC; naive values are treated as UTC."""

    if dt.tzinfo is None or dt.utcoffset() is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


class PitRowModel(BaseModel):
    """Base PIT schema with a canonical UTC contract for availability timestamps."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("available_at", mode="after", check_fields=False)
    @classmethod
    def normalize_available_at(cls, value: datetime) -> datetime:
        return ensure_utc(value)


class CorporateActionKind(StrEnum):
    SPLIT = "split"
    CASH_DIVIDEND = "cash_dividend"
    TICKER_CHANGE = "ticker_change"
    MERGER = "merger"
    DELISTING = "delisting"


class AssetMasterRow(PitRowModel):
    """Identifier history and listing metadata for a single asset at a PIT snapshot."""

    asset_id: str
    symbol: str
    primary_exchange: str | None = None
    listing_date: date | None = None
    delisting_date: date | None = None
    replaces_asset_id: str | None = Field(
        default=None,
        description="Prior asset_id superseded by this row (ticker lineage).",
    )
    replaced_by_asset_id: str | None = Field(
        default=None,
        description="Successor asset_id if this identifier is retired.",
    )
    snapshot_date: date = Field(
        description="Session date this master snapshot is keyed to for PIT replay.",
    )
    available_at: datetime = Field(
        description=(
            "When this master row may first appear in a leakage-safe query. "
            "Naive timestamps are treated as UTC and stored in UTC."
        ),
    )

    @model_validator(mode="after")
    def available_within_snapshot_session(self) -> AssetMasterRow:
        if ensure_utc(self.available_at) > utc_end_of_day(self.snapshot_date):
            msg = "available_at exceeds PIT visibility bound for snapshot_date"
            raise ValueError(msg)
        return self


class DailyBarRow(PitRowModel):
    asset_id: str
    session_date: date
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    dollar_volume: Decimal
    available_at: datetime = Field(
        description=(
            "When this bar may first be returned at session_date close. "
            "Naive timestamps are treated as UTC and stored in UTC."
        ),
    )

    @model_validator(mode="after")
    def available_within_session_day(self) -> DailyBarRow:
        if ensure_utc(self.available_at) > utc_end_of_day(self.session_date):
            msg = "available_at exceeds PIT visibility bound for session_date"
            raise ValueError(msg)
        return self


class CorporateActionRow(PitRowModel):
    asset_id: str
    action_type: CorporateActionKind
    effective_date: date
    ex_date: date | None = None
    split_from: Decimal | None = None
    split_to: Decimal | None = None
    dividend_amount: Decimal | None = None
    new_symbol: str | None = None
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Vendor-specific loose payload for action details not yet promoted "
            "into canonical fields."
        ),
    )
    available_at: datetime = Field(
        description=(
            "When this corporate action row may first be returned PIT-safe. "
            "Naive timestamps are treated as UTC and stored in UTC."
        ),
    )

    @model_validator(mode="after")
    def available_within_effective_day(self) -> CorporateActionRow:
        if ensure_utc(self.available_at) > utc_end_of_day(self.effective_date):
            msg = "available_at exceeds PIT visibility bound for effective_date"
            raise ValueError(msg)
        return self


class FundamentalRow(PitRowModel):
    """Point-in-time fundamentals gated by available_at."""

    asset_id: str
    fiscal_period_start: date | None = None
    fiscal_period_end: date
    revenue: Decimal | None = None
    net_income: Decimal | None = None
    shares_outstanding: Decimal | None = None
    available_at: datetime = Field(
        description=(
            "Vendor/filed availability for this fundamental snapshot. "
            "Naive timestamps are treated as UTC and stored in UTC."
        ),
    )

    @model_validator(mode="after")
    def available_not_before_period_end(self) -> FundamentalRow:
        if ensure_utc(self.available_at).date() < self.fiscal_period_end:
            msg = "available_at precedes fiscal_period_end"
            raise ValueError(msg)
        return self
