"""Tool input schemas for the OpenEnv environment surface."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class NoInput(_FrozenModel):
    """Empty schema used for zero-argument environment tools."""


class _GatedDateInput(_FrozenModel):
    """Mixin for view tools that accept an optional as_of_date.

    When ``as_of_date`` is provided, the runtime hard-refuses requests with
    ``as_of_date > current_simulation_date`` via :func:`tradebench.environment
    .date_gate.check_date_bound` — the tool returns a structured
    ``lookahead_violation`` error rather than the requested data. Omitting
    the field keeps the today-only default behavior.
    """

    as_of_date: date | None = Field(
        default=None,
        description=(
            "Optional past-or-current date for the query. Future dates are "
            "rejected with a lookahead_violation error."
        ),
    )


class ViewUniverseInput(_GatedDateInput):
    pass


class ViewTimeInput(NoInput):
    pass


class ViewPortfolioInput(NoInput):
    pass


class ViewOrdersInput(NoInput):
    pass


class ViewConstraintsInput(NoInput):
    pass


class AdvanceDayInput(NoInput):
    pass


class ViewEpisodeMetricsInput(_GatedDateInput):
    pass


class ConvictionInput(_FrozenModel):
    asset_id: str
    weight: Decimal


class RecordDecisionInput(_FrozenModel):
    regime_label: str
    edge_summary: str
    intended_exposure: Decimal
    top_convictions: list[ConvictionInput] = Field(default_factory=list)
    uncertainty: Literal["low", "medium", "high"]
    reasoning: str | None = Field(
        default=None,
        description=(
            "Optional free-text justification for the decision. Scanned by the "
            "rules-clause verifier alongside edge_summary; phrases that recall "
            "specific historical outcomes trip r_rules=-1.0 on the next "
            "advance_day."
        ),
    )


class PlaceOrderInput(_FrozenModel):
    client_order_id: str
    asset_id: str
    side: Literal["buy", "sell"]
    quantity: int = Field(gt=0)


class CancelOrderInput(_FrozenModel):
    client_order_id: str


class SandboxExecInput(_FrozenModel):
    """Run a command inside the episode sandbox and return stdout/stderr."""

    command: list[str] = Field(
        min_length=1,
        description="Command and arguments to execute inside the sandbox.",
    )
    env: dict[str, str] = Field(
        default_factory=dict,
        description="Extra environment variables for the sandboxed process.",
    )
    timeout_seconds: int | None = Field(
        default=None,
        gt=0,
        description="Optional hard timeout for the sandboxed process.",
    )


__all__ = [
    "AdvanceDayInput",
    "CancelOrderInput",
    "ConvictionInput",
    "NoInput",
    "PlaceOrderInput",
    "RecordDecisionInput",
    "ViewConstraintsInput",
    "ViewEpisodeMetricsInput",
    "ViewOrdersInput",
    "ViewPortfolioInput",
    "ViewTimeInput",
    "SandboxExecInput",
    "ViewUniverseInput",
]
