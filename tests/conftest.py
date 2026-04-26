"""Shared pytest fixtures for TradeBench tests."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from tradebench.data.catalog import DatasetCatalog
from tradebench.data.query import PitQueryService
from tradebench.data.sample_dataset import SAMPLE_DATASET_VERSION, write_sample_dataset


@pytest.fixture
def sample_dataset_root(tmp_path: Path) -> Path:
    """Temporary dataset root containing the deterministic sample catalog."""

    write_sample_dataset(tmp_path)
    return tmp_path


@pytest.fixture
def sample_catalog(sample_dataset_root: Path) -> DatasetCatalog:
    return DatasetCatalog(sample_dataset_root, SAMPLE_DATASET_VERSION)


@pytest.fixture
def pit_query_service(sample_catalog: DatasetCatalog) -> Iterator[PitQueryService]:
    with PitQueryService(sample_catalog) as service:
        yield service
