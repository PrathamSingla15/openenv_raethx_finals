"""Progressive filesystem materialization for date-gated sandbox mounts.

Implementation is **copy-on-reset**: at episode start we walk
``source_root`` and copy only those files whose ``available_at <= current_date``
into ``dest_root``. On each :func:`refresh_allowed_files` call (invoked from
``advance_day``), we copy newly-available files. No files are ever removed;
the sandbox sees only the past, never the future.

Per-file availability is inferred from partition path segments
(``/yyyy-mm-dd/`` or ``/yyyy/mm/dd/``). Files without an inferable partition
date fall through as **fail-open** — PIT enforcement still happens at the
SQL layer via per-row ``available_at`` filters. Callers that want stricter
behavior can pass a ``file_metadata`` callable for full control.
"""

from __future__ import annotations

import logging
import re
import shutil
from collections.abc import Callable
from datetime import date, datetime
from pathlib import Path

__all__ = [
    "materialize_allowed_files",
    "refresh_allowed_files",
]


log = logging.getLogger(__name__)


_PARTITION_RE = re.compile(
    r"(?:^|/)(?P<y>\d{4})[-/](?P<m>\d{2})[-/](?P<d>\d{2})(?:/|$)"
)


def _infer_partition_date(path: Path, source_root: Path) -> date | None:
    """Best-effort partition-date extraction from a path relative to root.

    Matches ``/yyyy-mm-dd/`` and ``/yyyy/mm/dd/`` segments. Returns
    ``None`` when no partition date is present — caller decides the
    fail-open vs fail-closed policy.
    """

    rel = str(path.relative_to(source_root)).replace("\\", "/")
    match = _PARTITION_RE.search("/" + rel)
    if match is None:
        return None
    try:
        return date(int(match["y"]), int(match["m"]), int(match["d"]))
    except ValueError:
        return None


def _coerce_date(value: date | str | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _should_copy(
    path: Path,
    source_root: Path,
    current: date,
    file_metadata: Callable[[Path], date | None] | None,
) -> bool:
    if file_metadata is not None:
        available = file_metadata(path)
    else:
        available = _infer_partition_date(path, source_root)

    if available is None:
        # Fail-open: unknown availability is included. PIT enforcement still
        # happens at the SQL layer via per-row ``available_at`` filters.
        log.debug(
            "progressive_fs: no available_at inferred for %s; copying anyway",
            path,
        )
        return True
    return available <= current


def _copy_file(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def materialize_allowed_files(
    source_root: Path,
    dest_root: Path,
    current_date: date | str | datetime,
    *,
    file_metadata: Callable[[Path], date | None] | None = None,
) -> list[Path]:
    """Copy files with ``available_at <= current_date`` into ``dest_root``.

    Returns the list of destination paths actually written (new copies
    only). Existing destinations are overwritten to reflect any source
    changes — callers who want idempotent refresh-only behavior should
    use :func:`refresh_allowed_files`.
    """

    src_root = source_root.expanduser().resolve()
    dst_root = dest_root.expanduser().resolve()
    if not src_root.is_dir():
        msg = f"source_root must be an existing directory: {src_root}"
        raise ValueError(msg)
    dst_root.mkdir(parents=True, exist_ok=True)
    current = _coerce_date(current_date)

    copied: list[Path] = []
    for src in sorted(p for p in src_root.rglob("*") if p.is_file()):
        if not _should_copy(src, src_root, current, file_metadata):
            continue
        rel = src.relative_to(src_root)
        dst = dst_root / rel
        _copy_file(src, dst)
        copied.append(dst)
    log.info(
        "progressive_fs: materialized %d files under %s (as of %s)",
        len(copied),
        dst_root,
        current.isoformat(),
    )
    return copied


def refresh_allowed_files(
    source_root: Path,
    dest_root: Path,
    new_current_date: date | str | datetime,
    file_metadata: Callable[[Path], date | None] | None = None,
) -> list[Path]:
    """Incrementally add newly-available files after an ``advance_day``.

    Files already present in ``dest_root`` are left untouched. Only
    sources that (a) satisfy the availability predicate at the new date
    *and* (b) don't yet exist at the destination are copied in.
    """

    src_root = source_root.expanduser().resolve()
    dst_root = dest_root.expanduser().resolve()
    if not src_root.is_dir():
        msg = f"source_root must be an existing directory: {src_root}"
        raise ValueError(msg)
    dst_root.mkdir(parents=True, exist_ok=True)
    current = _coerce_date(new_current_date)

    added: list[Path] = []
    for src in sorted(p for p in src_root.rglob("*") if p.is_file()):
        if not _should_copy(src, src_root, current, file_metadata):
            continue
        rel = src.relative_to(src_root)
        dst = dst_root / rel
        if dst.exists():
            continue
        _copy_file(src, dst)
        added.append(dst)
    log.info(
        "progressive_fs: refreshed +%d files under %s (as of %s)",
        len(added),
        dst_root,
        current.isoformat(),
    )
    return added


if __name__ == "__main__":
    # Smoke test: 3 dated files under a tmpdir; verify only past/present
    # files reach dest.
    import tempfile

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    with tempfile.TemporaryDirectory() as root_str:
        root = Path(root_str)
        src = root / "src"
        dst = root / "dst"
        past = src / "2020-01-02" / "bars.parquet"
        today = src / "2020-01-15" / "bars.parquet"
        future = src / "2020-02-10" / "bars.parquet"
        for p in (past, today, future):
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"x")

        written = materialize_allowed_files(src, dst, date(2020, 1, 15))
        names = sorted(p.name for p in written)
        print("initial copy:", [str(p.relative_to(dst)) for p in written])
        assert len(written) == 2, written
        assert (dst / "2020-01-02" / "bars.parquet").exists()
        assert (dst / "2020-01-15" / "bars.parquet").exists()
        assert not (dst / "2020-02-10" / "bars.parquet").exists()

        added = refresh_allowed_files(src, dst, date(2020, 2, 10))
        print("refresh:", [str(p.relative_to(dst)) for p in added])
        assert len(added) == 1, added
        assert (dst / "2020-02-10" / "bars.parquet").exists()

        noop = refresh_allowed_files(src, dst, date(2020, 2, 10))
        print("noop refresh:", noop)
        assert noop == []
        print("progressive_fs smoke test OK")
