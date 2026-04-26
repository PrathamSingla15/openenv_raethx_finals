"""Deterministic transaction cost and slippage model (Decimal)."""

from __future__ import annotations

from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CostModel(BaseModel):
    """
    Per-fill costs:

    - ``commission_floor``: fixed currency charge per executed order (one side).
    - ``fee_bps``: proportional fee on executed notional
      (``notional * fee_bps / 10_000``).
    - ``slippage_bps_cap``: upper bound on slippage in basis points.

    Slippage scales with **participation** ``notional / trailing_dollar_volume``:

    ``slippage_bps = min(slippage_bps_cap, participation * 10_000)``

    So 1% of trailing dollar volume as notional implies 100 bps of slippage, capped.
    If ``trailing_dollar_volume <= 0``, slippage is set to ``slippage_bps_cap``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    commission_floor: Decimal = Field(ge=0)
    fee_bps: Decimal = Field(ge=0)
    slippage_bps_cap: Decimal = Field(ge=0)


def compute_slippage_bps(
    model: CostModel,
    *,
    order_notional: Decimal,
    trailing_dollar_volume: Decimal,
) -> Decimal:
    """Return nonnegative, cap-applied slippage in basis points."""

    if trailing_dollar_volume <= 0:
        return model.slippage_bps_cap
    participation = order_notional / trailing_dollar_volume
    raw_bps = participation * Decimal("10000")
    return min(model.slippage_bps_cap, raw_bps)


def compute_fee_total(model: CostModel, *, fill_notional: Decimal) -> Decimal:
    """Commission floor plus proportional bps fee on filled notional."""

    return model.commission_floor + fill_notional * model.fee_bps / Decimal("10000")


def fill_price_from_open(
    session_open: Decimal,
    side: Literal["buy", "sell"],
    slippage_bps: Decimal,
) -> Decimal:
    """
    Adjust the **regular-way open** on the execution session.

    Buys pay a higher effective price; sells receive a lower effective price:

    - buy: ``open * (1 + slippage_bps / 10_000)``
    - sell: ``open * (1 - slippage_bps / 10_000)``
    """

    slip = slippage_bps / Decimal("10000")
    if side == "buy":
        return session_open * (Decimal("1") + slip)
    return session_open * (Decimal("1") - slip)
