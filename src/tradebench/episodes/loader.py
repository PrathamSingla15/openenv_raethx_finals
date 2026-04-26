"""Deterministic enumeration and indexing of on-disk episode manifests."""

from __future__ import annotations

import os
from pathlib import Path

from tradebench.episodes.models import EpisodeManifest, manifest_digest

_DATASET_ROOT_ENV_VAR = "TRADEBENCH_DATASET_ROOT"


def _resolve_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").is_file():
            return parent

    msg = (
        "Could not resolve TradeBench project root from "
        f"{current}; no pyproject.toml found in parent directories."
    )
    raise RuntimeError(msg)


def _resolve_dataset_root() -> Path:
    override = os.environ.get(_DATASET_ROOT_ENV_VAR)
    if override:
        return Path(override).expanduser().resolve()
    from tradebench.data.sample_dataset import ensure_sample_dataset

    return ensure_sample_dataset(_resolve_project_root() / "datasets")


def _manifest_paths_under_catalog(dataset_root: Path) -> list[Path]:
    catalog = dataset_root / "catalog"
    if not catalog.is_dir():
        return []
    paths: list[Path] = []
    for version_dir in sorted(p for p in catalog.iterdir() if p.is_dir()):
        episode_dir = version_dir / "episode_manifests"
        if episode_dir.is_dir():
            paths.extend(sorted(episode_dir.glob("*.json")))
    return paths


def _manifest_sort_key(
    manifest: EpisodeManifest,
) -> tuple[str, str, str, str]:
    return (
        manifest.task_id,
        manifest.episode_start.isoformat(),
        manifest.episode_end.isoformat(),
        manifest_digest(manifest),
    )


def list_manifests(split: str) -> list[EpisodeManifest]:
    """List manifests for a split in stable order."""

    dataset_root = _resolve_dataset_root()
    selected: list[tuple[Path, EpisodeManifest]] = []
    for path in _manifest_paths_under_catalog(dataset_root):
        raw = path.read_text(encoding="utf-8")
        manifest = EpisodeManifest.model_validate_json(raw)
        if manifest.split.name == split:
            selected.append((path, manifest))
    selected.sort(key=lambda item: _manifest_sort_key(item[1]))
    return [m for _, m in selected]


def get_manifest(split: str, index: int) -> EpisodeManifest:
    manifests = list_manifests(split)
    return manifests[index]


def get_manifest_range(
    split: str,
    start: int,
    stop: int | None = None,
) -> list[EpisodeManifest]:
    return list_manifests(split)[start:stop]
