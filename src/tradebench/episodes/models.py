"""Episode manifest schemas and stable content addressing."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class SplitSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: Literal["debug", "train", "validation", "test"]


class ConstraintPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_single_name_weight: Decimal | None = Field(
        default=None,
        description="Cap on weight of any single issuer/name, if enforced.",
    )
    max_portfolio_turnover: Decimal | None = Field(
        default=None,
        description="Optional turnover budget expressed as a fraction of portfolio.",
    )
    max_gross_exposure: Decimal | None = Field(
        default=None,
        description="Optional cap on gross exposure (e.g. 1.0 long-only).",
    )


class ConvictionEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    asset_id: str
    weight: Decimal


class DecisionSnapshot(BaseModel):
    """Structured decision record captured before first order on a session date."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    regime_label: str
    edge_summary: str
    intended_exposure: Decimal
    top_convictions: list[ConvictionEntry] = Field(default_factory=list)
    uncertainty: Literal["low", "medium", "high"]


class EpisodeManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_version: str
    task_id: str
    split: SplitSpec
    warmup_start: date
    episode_start: date
    episode_end: date
    initial_cash: Decimal
    universe_asset_ids: list[str]
    cost_model_version: str
    scorer_version: str
    sandbox_image_digest: str
    dependency_lock_digest: str
    regime_labels: dict[str, str]
    constraint_policy: ConstraintPolicy | None = None


def manifest_digest(manifest: EpisodeManifest) -> str:
    """Return a stable SHA-256 hex digest of canonical JSON for the manifest."""

    payload: dict[str, Any] = manifest.model_dump(mode="json")
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
