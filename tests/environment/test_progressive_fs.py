"""Pytest coverage for the progressive filesystem materializer."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from tradebench.environment.progressive_fs import (
    materialize_allowed_files,
    refresh_allowed_files,
)


@pytest.fixture
def synthetic_tree(tmp_path: Path) -> tuple[Path, Path]:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    for partition in ("2020-01-02", "2020-01-15", "2020-02-10"):
        (src / partition).mkdir(parents=True)
        (src / partition / "bars.parquet").write_bytes(b"x")
    return src, dst


def test_materialize_only_includes_past_partitions(synthetic_tree) -> None:
    src, dst = synthetic_tree
    written = materialize_allowed_files(src, dst, date(2020, 1, 15))
    rel = sorted(p.relative_to(dst).as_posix() for p in written)
    assert "2020-01-02/bars.parquet" in rel
    assert "2020-01-15/bars.parquet" in rel
    assert "2020-02-10/bars.parquet" not in rel
    assert not (dst / "2020-02-10" / "bars.parquet").exists()


def test_refresh_adds_newly_available_files(synthetic_tree) -> None:
    src, dst = synthetic_tree
    materialize_allowed_files(src, dst, date(2020, 1, 15))
    added = refresh_allowed_files(src, dst, date(2020, 2, 10))
    rel = sorted(p.relative_to(dst).as_posix() for p in added)
    assert rel == ["2020-02-10/bars.parquet"]


def test_refresh_is_idempotent_with_no_new_files(synthetic_tree) -> None:
    src, dst = synthetic_tree
    materialize_allowed_files(src, dst, date(2020, 2, 10))
    again = refresh_allowed_files(src, dst, date(2020, 2, 10))
    assert again == []


def test_unpartitioned_files_fall_open(tmp_path: Path) -> None:
    """Files without an inferable partition date are copied with a warning.

    This is the documented fail-open policy in progressive_fs._should_copy
    so the existing flat-parquet sample dataset doesn't break.
    """

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    (src / "calendar.parquet").write_bytes(b"x")
    written = materialize_allowed_files(src, dst, date(2020, 1, 15))
    assert (dst / "calendar.parquet").exists()
    assert any(p.name == "calendar.parquet" for p in written)


def test_explicit_metadata_callback_takes_precedence(synthetic_tree) -> None:
    src, dst = synthetic_tree

    # Force "every file is dated 1900-01-01" → all should be copied at 2020-01-15.
    def force_old(path: Path) -> date:
        return date(1900, 1, 1)

    written = materialize_allowed_files(
        src,
        dst,
        date(2020, 1, 15),
        file_metadata=force_old,
    )
    assert len(written) == 3  # all 3 partitions admitted
