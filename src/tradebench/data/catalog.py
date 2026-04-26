"""Versioned dataset catalog layout and path resolution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetCatalogPaths:
    """Absolute paths for a single `datasets/catalog/<dataset_version>/` tree."""

    version_dir: Path
    asset_master: Path
    daily_bars_dir: Path
    corporate_actions: Path
    fundamentals_pti: Path
    calendar: Path
    episode_manifests_dir: Path

    @property
    def daily_bars_glob(self) -> str:
        """Glob string for DuckDB `read_parquet` over all daily bar shards."""

        return str(self.daily_bars_dir / "*.parquet")


class DatasetCatalog:
    """Discover Parquet tables under `dataset_root/catalog/<dataset_version>/`."""

    def __init__(self, dataset_root: Path, dataset_version: str) -> None:
        self.dataset_root = dataset_root.expanduser().resolve()
        self.dataset_version = dataset_version
        self.version_dir = self.dataset_root / "catalog" / dataset_version

    def paths(self) -> DatasetCatalogPaths:
        return DatasetCatalogPaths(
            version_dir=self.version_dir,
            asset_master=self.version_dir / "asset_master.parquet",
            daily_bars_dir=self.version_dir / "daily_bars",
            corporate_actions=self.version_dir / "corporate_actions.parquet",
            fundamentals_pti=self.version_dir / "fundamentals_pti.parquet",
            calendar=self.version_dir / "calendar.parquet",
            episode_manifests_dir=self.version_dir / "episode_manifests",
        )

    def validate_layout(self) -> None:
        """Ensure required files exist (raises FileNotFoundError if not)."""

        p = self.paths()
        for label, path in (
            ("asset_master", p.asset_master),
            ("corporate_actions", p.corporate_actions),
            ("fundamentals_pti", p.fundamentals_pti),
            ("calendar", p.calendar),
        ):
            if not path.is_file():
                msg = f"Missing {label} parquet: {path}"
                raise FileNotFoundError(msg)
        if not p.daily_bars_dir.is_dir():
            msg = f"Missing daily_bars directory: {p.daily_bars_dir}"
            raise FileNotFoundError(msg)
        if not any(p.daily_bars_dir.glob("*.parquet")):
            msg = f"No parquet shards under {p.daily_bars_dir}"
            raise FileNotFoundError(msg)
        if not p.episode_manifests_dir.is_dir():
            msg = f"Missing episode_manifests directory: {p.episode_manifests_dir}"
            raise FileNotFoundError(msg)
