"""Task-tier to EpisodeManifest mapping.

Three tiers, three roles:

- ``t1`` -> ``tier_t1`` (60-bar window, 5 assets, debug). Smoke-test tier
  for harness validation and prompt iteration. Not part of the train/test
  split.
- ``train`` -> ``tier_train`` (252-bar window, 10 shared aliases, study
  packet). At reset the full window is delivered in-context; the agent
  derives a strategy and emits a single ``record_decision``. No per-bar
  rollout, no reward.
- ``test`` -> ``tier_test`` (120-bar window, same 10 aliases, held-out
  rollout). Calendar-adjacent to ``train`` (starts the trading day after
  ``train`` ends). The agent walks it bar by bar; composite reward emits
  per ``advance_day``. This is what we score.

Manifests are seeded by ``tradebench.data.sample_dataset.write_sample_dataset``
into ``datasets/catalog/<version>/episode_manifests/<task_id>.json``.
"""

from __future__ import annotations

from typing import Literal, cast

from tradebench.episodes.loader import list_manifests
from tradebench.episodes.models import EpisodeManifest

TierId = Literal["t1", "train", "test"]

TIER_IDS: tuple[TierId, ...] = ("t1", "train", "test")

TIER_TO_TASK_ID: dict[TierId, str] = {
    "t1": "tier_t1",
    "train": "tier_train",
    "test": "tier_test",
}

TIER_TO_SPLIT: dict[TierId, str] = {
    "t1": "debug",
    "train": "train",
    "test": "test",
}

TIER_DESCRIPTIONS: dict[TierId, dict[str, str]] = {
    "t1": {
        "name": "T1 - Smoke",
        "regime": "Short rollout, debug",
        "horizon": "60 bars (~3 months)",
        "intent": (
            "Smoke-test tier. Used for harness validation and prompt iteration "
            "(~10-15 min wall time per rollout). Not part of the train/test split."
        ),
    },
    "train": {
        "name": "Train - Study packet",
        "regime": "1-year study window, real-derived",
        "horizon": "252 bars (~1 year)",
        "intent": (
            "The full window is delivered in-context at episode reset (OHLCV + "
            "summary stats + correlation matrix). The agent derives a strategy "
            "from this packet and emits a single record_decision summarising it. "
            "No per-bar rollout, no reward."
        ),
    },
    "test": {
        "name": "Test - Held-out rollout",
        "regime": "Out-of-sample, calendar-adjacent to train",
        "horizon": "120 bars (~6 months)",
        "intent": (
            "Held-out single-shot evaluation. Walked one bar at a time using "
            "the strategy derived during train. Composite reward emits per "
            "advance_day; this is the score that matters."
        ),
    },
}


def list_tiers() -> list[TierId]:
    """Return the canonical tier id list in canonical order."""

    return list(TIER_IDS)


def load_tier(tier_id: TierId | str) -> EpisodeManifest:
    """Return the :class:`EpisodeManifest` whose ``task_id`` matches ``tier_id``.

    Searches every split (debug / train / validation / test) for a manifest
    with ``task_id == TIER_TO_TASK_ID[tier]``. Raises :class:`RuntimeError`
    with a remediation hint when no match is found, which usually means the
    sample-dataset writer hasn't run.
    """

    if tier_id not in TIER_TO_TASK_ID:
        msg = f"Unknown task tier {tier_id!r}; expected one of {TIER_IDS}"
        raise ValueError(msg)
    tier = cast(TierId, tier_id)
    target_task_id = TIER_TO_TASK_ID[tier]
    for split in ("debug", "train", "validation", "test"):
        for manifest in list_manifests(split):
            if manifest.task_id == target_task_id:
                return manifest
    msg = (
        f"No manifest with task_id={target_task_id!r} found for tier {tier}. "
        "Delete ``datasets/`` and re-run any path that calls "
        "``ensure_sample_dataset`` to seed the catalog, or point "
        "TRADEBENCH_DATASET_ROOT at a catalog that contains the "
        "purpose-built tier manifests."
    )
    raise RuntimeError(msg)


def describe_tier(tier_id: TierId | str) -> dict[str, str]:
    """Return the human-facing description of the requested tier."""

    if tier_id not in TIER_DESCRIPTIONS:
        msg = f"Unknown task tier {tier_id!r}; expected one of {TIER_IDS}"
        raise ValueError(msg)
    tier = cast(TierId, tier_id)
    return dict(TIER_DESCRIPTIONS[tier])


__all__ = [
    "TierId",
    "TIER_IDS",
    "TIER_TO_TASK_ID",
    "TIER_TO_SPLIT",
    "TIER_DESCRIPTIONS",
    "list_tiers",
    "load_tier",
    "describe_tier",
]
