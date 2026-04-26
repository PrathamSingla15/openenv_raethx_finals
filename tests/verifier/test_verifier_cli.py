"""Smoke tests for the TradeBench verifier orchestrator + CLI.

We don't unit-test each check in isolation here — instead we drive the
full ``run_verifier`` for every tier and assert the aggregate result.
That catches integration regressions (e.g., a future change to
TradeObservation that breaks the schema check, or a new look-ahead
leak class that the FS audit misses) without overfitting to per-check
implementation details.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tradebench.verifier import (
    CheckSeverity,
    format_report,
    run_verifier,
)
from tradebench.verifier.cli import main as verifier_main
from tradebench.verifier.leak import (
    _infer_partition_date,
    check_filesystem_audit,
)


def _force_local_sandbox(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRADEBENCH_SANDBOX_PROVIDER", "local")


# Train tier is a one-shot study episode (no per-bar rollout, no reward), so
# the verifier's conformance / replay / determinism checks don't apply to it
# the way they apply to a trading rollout. Only the trading tiers are verified.
@pytest.mark.parametrize("tier", ["t1", "test"])
def test_run_verifier_passes_on_each_tier(
    tier: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_local_sandbox(monkeypatch)
    report = run_verifier(tier=tier, seed=42, max_bars=3)
    assert not report.has_errors, format_report(report)
    # Every named check should be present
    names = {r.name for r in report.results}
    assert "conformance.observation" in names
    assert "conformance.action_rejection" in names
    assert "leak.no_future_fit" in names
    assert "replay" in names
    assert "determinism" in names


def test_cli_entry_point_returns_zero_on_clean_episode(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _force_local_sandbox(monkeypatch)
    code = verifier_main(["--tier", "t1", "--seed", "7", "--max-bars", "2"])
    captured = capsys.readouterr()
    assert code == 0, captured.out + captured.err
    assert "RESULT: PASS" in captured.out


def test_filesystem_audit_flags_future_partition(tmp_path: Path) -> None:
    """A file in a future-dated partition must trip the FS audit."""

    from datetime import date

    sandbox = tmp_path / "data"
    sandbox.mkdir()
    (sandbox / "2099-01-15").mkdir()
    (sandbox / "2099-01-15" / "bar.parquet").write_bytes(b"x")
    (sandbox / "2020-01-15").mkdir()
    (sandbox / "2020-01-15" / "bar.parquet").write_bytes(b"x")

    result = check_filesystem_audit(
        sandbox_data_dir=sandbox,
        current_date=date(2020, 1, 15),
        step_index=0,
    )
    assert result.severity is CheckSeverity.FAIL
    assert "leaked_paths" in result.details
    assert any("2099-01-15" in p for p in result.details["leaked_paths"])


def test_partition_date_inference() -> None:
    root = Path("/tmp/x")
    assert _infer_partition_date(root / "2020-03-15" / "f.parquet", root) == __import__(
        "datetime",
    ).date(2020, 3, 15)
    assert _infer_partition_date(root / "2020" / "03" / "15" / "f.parquet", root) == __import__(
        "datetime",
    ).date(2020, 3, 15)
    assert _infer_partition_date(root / "f.parquet", root) is None
