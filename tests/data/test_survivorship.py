"""Delisted and renamed identifiers remain queryable under historical PIT bounds."""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

from tradebench.data.query import PitQueryService
from tradebench.data.sample_dataset import (
    ASSET_DEAD,
    ASSET_LIQUID,
    ASSET_RENAME_NEW,
    ASSET_RENAME_OLD,
    DELIST_DATE,
    RENAME_DATE,
    SAMPLE_DATASET_VERSION,
    SESSION_END,
    SESSION_START,
)
from tradebench.episodes.models import EpisodeManifest, SplitSpec


def _full_universe_manifest() -> EpisodeManifest:
    return EpisodeManifest(
        dataset_version=SAMPLE_DATASET_VERSION,
        task_id="survivorship_probe",
        split=SplitSpec(name="train"),
        warmup_start=date(2019, 6, 3),
        episode_start=SESSION_START,
        episode_end=SESSION_END,
        initial_cash=Decimal("100000"),
        universe_asset_ids=[
            ASSET_LIQUID,
            ASSET_DEAD,
            ASSET_RENAME_OLD,
            ASSET_RENAME_NEW,
        ],
        cost_model_version="v0",
        scorer_version="v0",
        sandbox_image_digest="sha256:surv",
        dependency_lock_digest="sha256:surv",
        regime_labels={"probe": "survivorship"},
    )


def test_rename_successor_not_in_universe_before_effective_date(
    pit_query_service: PitQueryService,
) -> None:
    manifest = _full_universe_manifest()
    before = RENAME_DATE - timedelta(days=1)
    universe = pit_query_service.load_universe(before, manifest)
    assert ASSET_RENAME_NEW not in universe
    assert ASSET_RENAME_OLD in universe


def test_rename_successor_joins_universe_after_effective_date(
    pit_query_service: PitQueryService,
) -> None:
    manifest = _full_universe_manifest()
    universe = pit_query_service.load_universe(date(2020, 2, 18), manifest)
    assert ASSET_RENAME_NEW in universe


def test_delisted_asset_stays_in_historical_universe_for_manifest(
    pit_query_service: PitQueryService,
) -> None:
    manifest = _full_universe_manifest()
    universe = pit_query_service.load_universe(DELIST_DATE, manifest)
    assert ASSET_DEAD in universe


def test_delisted_symbol_bars_remain_queryable_before_delist(
    pit_query_service: PitQueryService,
) -> None:
    last_session_before = date(2020, 3, 13)
    df = pit_query_service.get_bars(
        [ASSET_DEAD],
        end_date=last_session_before,
        lookback_days=10,
    )
    assert not df.empty
    assert df["session_date"].max() <= last_session_before
