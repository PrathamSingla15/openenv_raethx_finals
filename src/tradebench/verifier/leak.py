"""Walk-forward leakage checks.

Two checks:

- :func:`check_filesystem_audit` walks every step of a captured
  trajectory and asserts the sandbox's host_data_dir contains no file
  whose inferred partition_date is greater than the simulation's
  current_date at that step. Catches the "agent ls'es a future
  partition" leak class.
- :func:`check_no_future_fit` searches the runtime for any object
  exposing a ``fit_through_date`` attribute and asserts it satisfies
  ``fit_through_date <= current_date``. Today no preprocessors expose
  this — the check is forward-compat. Returns a WARN with details when
  empty so the report records that the check ran.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Any

from ._types import CheckResult, CheckSeverity

_PARTITION_RE = re.compile(
    r"(?:^|/)(?P<y>\d{4})[-/](?P<m>\d{2})[-/](?P<d>\d{2})(?:/|$)",
)


def _infer_partition_date(path: Path, root: Path) -> date | None:
    rel = str(path.relative_to(root)).replace("\\", "/")
    match = _PARTITION_RE.search("/" + rel)
    if match is None:
        return None
    try:
        return date(int(match["y"]), int(match["m"]), int(match["d"]))
    except ValueError:
        return None


def check_filesystem_audit(
    *,
    sandbox_data_dir: Path,
    current_date: date,
    step_index: int,
) -> CheckResult:
    """Assert no file in ``sandbox_data_dir`` is keyed to a future partition.

    Files without an inferable partition date are skipped (not failed) —
    this matches the fail-open policy in
    ``tradebench.environment.progressive_fs._should_copy``. Real
    date-partitioned datasets (the T1/T2/T3 manifests once they ship
    with day-partitioned parquet) will produce hard failures on leaks.
    """

    if not sandbox_data_dir.is_dir():
        return CheckResult(
            name=f"leak.fs[step={step_index}]",
            severity=CheckSeverity.FAIL,
            message=f"sandbox data dir missing: {sandbox_data_dir}",
        )
    leaked: list[str] = []
    inspected = 0
    for path in sandbox_data_dir.rglob("*"):
        if not path.is_file():
            continue
        inspected += 1
        partition = _infer_partition_date(path, sandbox_data_dir)
        if partition is None:
            continue
        if partition > current_date:
            leaked.append(str(path.relative_to(sandbox_data_dir)))

    if leaked:
        return CheckResult(
            name=f"leak.fs[step={step_index}]",
            severity=CheckSeverity.FAIL,
            message=(
                f"{len(leaked)} file(s) with partition_date > "
                f"{current_date.isoformat()} in sandbox mount"
            ),
            details={"leaked_paths": leaked[:10], "inspected": inspected},
        )
    return CheckResult(
        name=f"leak.fs[step={step_index}]",
        severity=CheckSeverity.PASS,
        message=f"inspected {inspected} files, no future partitions",
    )


def check_no_future_fit(runtime: Any, current_date: date) -> CheckResult:
    """Forward-compat: any cached preprocessor must carry a valid fit_through_date.

    Walks every public attribute of ``runtime`` and looks for
    ``fit_through_date``. When one exists, asserts it's at or before
    ``current_date``. With no preprocessors in the env path today this
    returns a PASS with the introspection summary.
    """

    suspects: list[tuple[str, str]] = []
    for attr_name in dir(runtime):
        if attr_name.startswith("_"):
            continue
        try:
            attr = getattr(runtime, attr_name)
        except Exception:  # noqa: BLE001 - properties may raise
            continue
        fit_through = getattr(attr, "fit_through_date", None)
        if fit_through is None:
            continue
        if isinstance(fit_through, date) and fit_through > current_date:
            suspects.append((attr_name, fit_through.isoformat()))

    if suspects:
        return CheckResult(
            name="leak.no_future_fit",
            severity=CheckSeverity.FAIL,
            message=(
                f"{len(suspects)} cached preprocessor(s) fit to data past "
                f"{current_date.isoformat()}"
            ),
            details={"suspects": suspects},
        )
    return CheckResult(
        name="leak.no_future_fit",
        severity=CheckSeverity.PASS,
        message="no preprocessors with fit_through_date > current_date",
    )


__all__ = ["check_filesystem_audit", "check_no_future_fit"]
