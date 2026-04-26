"""Read-model shapes for portfolio snapshots derived from the ledger."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Position(BaseModel):
    """Aggregated long position in one asset (integer shares, long-only)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    shares: int = Field(ge=0)
    cost_basis_total: Decimal = Field(
        default=Decimal("0"),
        description=(
            "Total cost basis for remaining shares "
            "(gross of fees embedded at lot acquisition)."
        ),
    )


class OpenOrder(BaseModel):
    """Working order and cash reservation for buy-side liquidity checks."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    client_order_id: str
    asset_id: str
    side: Literal["buy", "sell"]
    quantity: int = Field(gt=0)
    cash_reserved: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        description=(
            "Cash held aside on submit for buys; refunded on cancel/reject/fill."
        ),
    )


class PortfolioState(BaseModel):
    """Environment-owned portfolio snapshot after projecting ledger events."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cash: Decimal = Field(description="Settled cash after reservations and fills.")
    positions_by_asset: dict[str, Position] = Field(default_factory=dict)
    open_orders: dict[str, OpenOrder] = Field(
        default_factory=dict,
        description="Keyed by client_order_id.",
    )
    last_valuation: Decimal | None = Field(
        default=None,
        description=(
            "Last mark-to-market total portfolio value from DayAdvanced, if any."
        ),
    )
    realized_pnl: Decimal = Field(default=Decimal("0"))
    unrealized_pnl: Decimal = Field(default=Decimal("0"))
    last_session_date: date | None = Field(default=None)

    @staticmethod
    def initial() -> PortfolioState:
        """Empty episode portfolio before any cash or position events."""

        return PortfolioState(cash=Decimal("0"))
