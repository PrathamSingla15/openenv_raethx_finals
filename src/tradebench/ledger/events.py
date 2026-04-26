"""Append-only ledger event schemas (immutable, content-addressable replay)."""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from tradebench.episodes.models import DecisionSnapshot


def ensure_utc(dt: datetime) -> datetime:
    """Normalize timestamps to UTC; naive values are treated as UTC."""

    if dt.tzinfo is None or dt.utcoffset() is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


class LedgerEventBase(BaseModel):
    """Shared envelope for every append-only ledger row."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: str
    episode_id: str
    event_time: datetime

    @field_validator("event_time", mode="after")
    @classmethod
    def normalize_event_time(cls, value: datetime) -> datetime:
        return ensure_utc(value)


class DecisionRecorded(LedgerEventBase):
    kind: Literal["decision_recorded"] = "decision_recorded"
    snapshot: DecisionSnapshot


class OrderSubmitted(LedgerEventBase):
    kind: Literal["order_submitted"] = "order_submitted"
    client_order_id: str = Field(
        description=(
            "Client-supplied order key. (episode_id, client_order_id) is "
            "treated as a single-use idempotency key during projection."
        ),
    )
    asset_id: str
    side: Literal["buy", "sell"]
    quantity: int = Field(gt=0)
    cash_reserved: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        description="Cash to reserve on buy submit; ignored for sells.",
    )


class OrderRejected(LedgerEventBase):
    kind: Literal["order_rejected"] = "order_rejected"
    client_order_id: str


class OrderCancelled(LedgerEventBase):
    kind: Literal["order_cancelled"] = "order_cancelled"
    client_order_id: str


class OrderFilled(LedgerEventBase):
    kind: Literal["order_filled"] = "order_filled"
    client_order_id: str
    asset_id: str
    side: Literal["buy", "sell"]
    quantity: int = Field(gt=0)
    avg_fill_price: Decimal = Field(gt=0)
    fees: Decimal = Field(default=Decimal("0"), ge=0)


class DividendApplied(LedgerEventBase):
    kind: Literal["dividend_applied"] = "dividend_applied"
    asset_id: str
    cash_credited: Decimal = Field(
        ge=0,
        description="Total cash credited to the portfolio from this dividend.",
    )


class SplitApplied(LedgerEventBase):
    kind: Literal["split_applied"] = "split_applied"
    asset_id: str
    ratio: Decimal = Field(
        gt=0,
        description="Share multiplier (e.g. 2 for a 2-for-1 split).",
    )


class TickerChanged(LedgerEventBase):
    kind: Literal["ticker_changed"] = "ticker_changed"
    from_asset_id: str
    to_asset_id: str


class DelistingLiquidated(LedgerEventBase):
    kind: Literal["delisting_liquidated"] = "delisting_liquidated"
    asset_id: str
    quantity_liquidated: int = Field(ge=0)
    cash_proceeds: Decimal = Field(default=Decimal("0"), ge=0)
    fees: Decimal = Field(default=Decimal("0"), ge=0)


class DayAdvanced(LedgerEventBase):
    kind: Literal["day_advanced"] = "day_advanced"
    session_date: date
    portfolio_market_value: Decimal | None = Field(
        default=None,
        ge=0,
        description=(
            "Total mark-to-market portfolio value after the session, if reported."
        ),
    )
    unrealized_pnl: Decimal | None = Field(
        default=None,
        description="Optional explicit unrealized PnL snapshot from the environment.",
    )


LedgerEvent = Annotated[
    DecisionRecorded
    | OrderSubmitted
    | OrderRejected
    | OrderCancelled
    | OrderFilled
    | DividendApplied
    | SplitApplied
    | TickerChanged
    | DelistingLiquidated
    | DayAdvanced,
    Field(discriminator="kind"),
]
