"""Train/test pair invariants and study-packet shape."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest

from tradebench.data.sample_dataset import SAMPLE_DATASET_VERSION
from tradebench.episodes.loader import list_manifests
from tradebench.episodes.study_packet import build_study_packet


@pytest.fixture
def isolated_dataset(
    sample_dataset_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """Point ``list_manifests`` at the per-test sample dataset root."""

    monkeypatch.setenv("TRADEBENCH_DATASET_ROOT", str(sample_dataset_root))
    return sample_dataset_root


def _manifest_by_id(split: str, task_id: str):
    for manifest in list_manifests(split):
        if manifest.task_id == task_id:
            return manifest
    raise AssertionError(f"manifest {task_id!r} not found in split {split!r}")


def _next_weekday(d: date) -> date:
    nxt = d + timedelta(days=1)
    while nxt.weekday() >= 5:
        nxt = nxt + timedelta(days=1)
    return nxt


def test_train_and_test_share_universe(isolated_dataset: Path) -> None:
    """The same alias list (and same order) must appear in both manifests.

    A strategy derived from the train tier is only meaningful on the test
    tier when the alias-to-instrument mapping is identical across both.
    """
    train = _manifest_by_id("train", "tier_train")
    test = _manifest_by_id("test", "tier_test")
    assert train.universe_asset_ids == test.universe_asset_ids, (
        "train and test manifests must share the same universe (same aliases, "
        "same order)"
    )


def test_test_starts_immediately_after_train(isolated_dataset: Path) -> None:
    """Test calendar must start the next trading day after train ends.

    Same-asset continuity across the train/test boundary is the whole point
    of the unified bake; an embargo gap or overlap would silently break it.
    """
    train = _manifest_by_id("train", "tier_train")
    test = _manifest_by_id("test", "tier_test")
    expected_test_start = _next_weekday(train.episode_end)
    assert test.episode_start == expected_test_start, (
        f"test.episode_start {test.episode_start} should be the trading day "
        f"after train.episode_end {train.episode_end} ({expected_test_start})"
    )


def test_study_packet_within_token_budget(isolated_dataset: Path) -> None:
    """Study packet text must fit a ~12K-token budget at 252 bars / 10 assets.

    The packet rides in-context alongside the system prompt; if it grows
    above ~12K tokens it crowds out the model's reasoning headroom and the
    test rollout can't carry forward a useful strategy.
    """
    train = _manifest_by_id("train", "tier_train")
    bars_path = (
        isolated_dataset
        / "catalog"
        / SAMPLE_DATASET_VERSION
        / "daily_bars"
        / "part-000.parquet"
    )
    if not bars_path.is_file():
        pytest.skip("daily_bars parquet missing — sample catalog not seeded")
    packet = build_study_packet(train, bars_parquet_path=bars_path)
    assert packet.bars_count == 252, (
        f"expected 252 bars in train window, got {packet.bars_count}"
    )
    assert len(packet.universe) == 10, (
        f"expected 10-asset shared universe, got {len(packet.universe)}"
    )
    approx_tokens = len(packet.text) // 4
    assert approx_tokens < 12_000, (
        f"study packet ~{approx_tokens} tokens exceeds 12K budget; "
        f"trim head/tail or per-asset stat columns"
    )
