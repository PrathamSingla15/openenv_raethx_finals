"""Shared dataclasses for the TradeBench verifier."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class CheckSeverity(StrEnum):
    """How a verifier check's result should be interpreted by the CLI."""

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


@dataclass(frozen=True)
class CheckResult:
    """One row of the verifier report — the outcome of a named check."""

    name: str
    severity: CheckSeverity
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class VerifierReport:
    """Aggregate result of running every verifier check on one episode."""

    tier: str
    seed: int
    results: list[CheckResult] = field(default_factory=list)

    def append(self, result: CheckResult) -> None:
        self.results.append(result)

    @property
    def has_errors(self) -> bool:
        return any(r.severity is CheckSeverity.FAIL for r in self.results)

    @property
    def has_warnings(self) -> bool:
        return any(r.severity is CheckSeverity.WARN for r in self.results)


__all__ = ["CheckResult", "CheckSeverity", "VerifierReport"]
