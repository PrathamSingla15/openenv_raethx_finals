"""Scalar-tolerance determinism check.

Two complete episodes with the same seed should produce equivalent
trajectories: terminal portfolio value within ``1e-6``, identical action
sequence and event count. We deliberately do not assert byte-identical event
logs because BLAS thread count, dict iteration, and float summation order
can break that guarantee under any concurrency.
"""

from __future__ import annotations

from typing import Any

from ._types import CheckResult, CheckSeverity

_TERMINAL_VALUE_TOL = 1e-6


def check_determinism(
    *,
    run_episode: Any,
    seed: int,
) -> CheckResult:
    """Run ``run_episode(seed)`` twice and compare scalar-tolerance summaries.

    ``run_episode`` is a zero-arg-after-binding callable returning a dict
    with ``final_value: float``, ``action_log: tuple[str, ...]``,
    ``event_count: int``. The CLI wires this via ``functools.partial``.
    """

    a = run_episode(seed)
    b = run_episode(seed)

    delta_value = abs(float(a["final_value"]) - float(b["final_value"]))
    actions_equal = tuple(a["action_log"]) == tuple(b["action_log"])
    event_count_equal = int(a["event_count"]) == int(b["event_count"])

    failures: list[str] = []
    if delta_value > _TERMINAL_VALUE_TOL:
        failures.append(
            f"|V_run1 - V_run2| = {delta_value:.3e} exceeds tolerance "
            f"{_TERMINAL_VALUE_TOL:.0e}",
        )
    if not actions_equal:
        failures.append("action_log diverged between runs")
    if not event_count_equal:
        failures.append(
            f"event_count differs: {a['event_count']} vs {b['event_count']}",
        )

    if failures:
        return CheckResult(
            name="determinism",
            severity=CheckSeverity.FAIL,
            message="; ".join(failures),
            details={
                "delta_value": delta_value,
                "actions_equal": actions_equal,
                "event_count_equal": event_count_equal,
            },
        )
    return CheckResult(
        name="determinism",
        severity=CheckSeverity.PASS,
        message=(
            f"|ΔV| = {delta_value:.3e} < {_TERMINAL_VALUE_TOL:.0e}; "
            f"{a['event_count']} events; {len(a['action_log'])} actions"
        ),
    )


__all__ = ["check_determinism"]
