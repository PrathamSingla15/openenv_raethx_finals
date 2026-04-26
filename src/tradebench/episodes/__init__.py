"""Episode manifests and deterministic manifest loading."""

from __future__ import annotations

import importlib

from tradebench.episodes.models import (
    ConstraintPolicy,
    DecisionSnapshot,
    EpisodeManifest,
    SplitSpec,
    manifest_digest,
)


def get_manifest(split: str, index: int) -> EpisodeManifest:
    from tradebench.episodes.loader import get_manifest as _get_manifest

    return _get_manifest(split, index)


def get_manifest_range(
    split: str,
    start: int,
    stop: int | None = None,
) -> list[EpisodeManifest]:
    from tradebench.episodes.loader import get_manifest_range as _get_manifest_range

    return _get_manifest_range(split, start, stop)


def list_manifests(split: str) -> list[EpisodeManifest]:
    from tradebench.episodes.loader import list_manifests as _list_manifests

    return _list_manifests(split)


def __getattr__(name: str) -> object:
    if name == "loader":
        return importlib.import_module(".loader", __name__)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)

__all__ = [
    "ConstraintPolicy",
    "DecisionSnapshot",
    "EpisodeManifest",
    "SplitSpec",
    "get_manifest",
    "get_manifest_range",
    "loader",
    "list_manifests",
    "manifest_digest",
]
