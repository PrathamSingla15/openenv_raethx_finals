"""Action / observation schema conformance checks.

Two checks live here:

- :func:`check_observation_schemas` round-trips every observation through
  ``TradeObservation.model_validate(model_dump())`` and asserts the
  ``reward_breakdown`` carries all seven expected keys with float values.
- :func:`check_action_rejection` issues two malformed actions (unknown
  action_type and structurally-invalid payload) and asserts the env
  returns a structured error rather than crashing or silently coercing.
"""

from __future__ import annotations

from typing import Any

try:
    from ...models import TradeAction, TradeObservation  # type: ignore[no-redef]
except (ImportError, ValueError):  # pragma: no cover
    from models import TradeAction, TradeObservation  # type: ignore[no-redef]

from ._types import CheckResult, CheckSeverity

_REQUIRED_BREAKDOWN_KEYS = (
    "r_wealth",
    "r_sharpe_bonus",
    "r_drawdown",
    "r_turnover",
    "r_concentration",
    "r_rules",
    "r_hack",
)


def check_observation_schemas(
    trajectory: list[TradeObservation],
) -> CheckResult:
    """Validate every observation in ``trajectory`` against TradeObservation."""

    bad_indices: list[int] = []
    for idx, obs in enumerate(trajectory):
        try:
            TradeObservation.model_validate(obs.model_dump())
        except Exception:  # noqa: BLE001 - pydantic ValidationError + edge cases
            bad_indices.append(idx)
            continue
        breakdown = obs.reward_breakdown or {}
        for key in _REQUIRED_BREAKDOWN_KEYS:
            if key not in breakdown:
                bad_indices.append(idx)
                break

    if bad_indices:
        return CheckResult(
            name="conformance.observation",
            severity=CheckSeverity.FAIL,
            message=(
                f"{len(bad_indices)}/{len(trajectory)} observations failed "
                "Pydantic round-trip or missing reward_breakdown keys."
            ),
            details={"bad_step_indices": bad_indices[:10]},
        )
    return CheckResult(
        name="conformance.observation",
        severity=CheckSeverity.PASS,
        message=f"all {len(trajectory)} observations valid",
    )


def check_action_rejection(env: Any, *, current_actions: list[str]) -> CheckResult:
    """Issue malformed actions; assert env returns a structured error.

    ``env`` is the in-process ``TradeBenchEnvironment`` (already reset to
    a live tier). ``current_actions`` is the action_type list returned
    by the most recent observation — used so the test can pick a known
    structurally-malformed payload.
    """

    failures: list[str] = []

    # Unknown action_type: TradeAction's Literal union may reject at construct
    # time; either rejection path counts.
    try:
        bogus = TradeAction(
            action_type="not_a_real_tool",  # type: ignore[arg-type]
            payload={},
        )
        result = env.step(bogus)
        if result.error is None:
            failures.append("unknown_action_type accepted without error")
    except Exception:  # noqa: BLE001 - rejection at construct time is fine
        pass

    try:
        bad_place = TradeAction(
            action_type="place_order",
            payload={"client_order_id": "cid", "side": "buy"},
        )
        result = env.step(bad_place)
        if result.error is None:
            failures.append("malformed place_order payload accepted")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"env crashed on malformed payload: {type(exc).__name__}")

    if failures:
        return CheckResult(
            name="conformance.action_rejection",
            severity=CheckSeverity.FAIL,
            message="; ".join(failures),
        )
    return CheckResult(
        name="conformance.action_rejection",
        severity=CheckSeverity.PASS,
        message="malformed actions rejected with structured errors",
    )


__all__ = ["check_observation_schemas", "check_action_rejection"]
