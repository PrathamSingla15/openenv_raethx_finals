"""TradeBench verifier — assert the env's load-bearing claims.

Public API:

- :func:`run_verifier` — drive an in-process env through one tier and run
  every check; returns a :class:`VerifierReport`.
- :func:`format_report` — render a report as the CLI-style PASS/FAIL table.

The package's ``__main__`` exposes the same orchestrator as a CLI:
``python -m tradebench.verifier --tier t1 --seed 42``.
"""

from .cli import format_report, run_verifier
from ._types import CheckResult, CheckSeverity, VerifierReport

__all__ = [
    "CheckResult",
    "CheckSeverity",
    "VerifierReport",
    "format_report",
    "run_verifier",
]
