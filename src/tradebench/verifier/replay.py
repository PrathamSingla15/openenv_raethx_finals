"""Reward-replay equality check.

Re-runs ``compute_composite_reward`` over the captured per-step inputs
and asserts each component matches the live-run breakdown within float
tolerance. Catches stateful-corruption bugs in the runtime's reward
buffers that would not fail a determinism check (which only compares
endpoints, not per-step components).
"""

from __future__ import annotations

import math
from typing import Any

from tradebench.rewards.composite import compute_composite_reward

from ._types import CheckResult, CheckSeverity

# r_wealth uses math.log and may differ by ULPs; other components match exactly.
_R_WEALTH_TOL = 1e-9


def check_reward_replay(
    *,
    captured_inputs: list[dict[str, Any]],
    captured_breakdowns: list[dict[str, float]],
) -> CheckResult:
    """Replay each step's reward and assert equality with the live capture.

    ``captured_inputs[i]`` is the kwargs dict passed to
    ``compute_composite_reward`` at advance_day step ``i``;
    ``captured_breakdowns[i]`` is ``RewardBreakdown.to_dict()`` from the
    live run. Both lists must align step-for-step.
    """

    if len(captured_inputs) != len(captured_breakdowns):
        return CheckResult(
            name="replay",
            severity=CheckSeverity.FAIL,
            message=(
                f"len(captured_inputs)={len(captured_inputs)} != "
                f"len(captured_breakdowns)={len(captured_breakdowns)}"
            ),
        )

    bad: list[tuple[int, str, float, float]] = []
    max_wealth_delta = 0.0
    for idx, (inputs, expected) in enumerate(
        zip(captured_inputs, captured_breakdowns, strict=True),
    ):
        replayed = compute_composite_reward(**inputs).to_dict()
        for key, expected_value in expected.items():
            replay_value = replayed.get(key, 0.0)
            if key == "r_wealth":
                delta = abs(replay_value - expected_value)
                max_wealth_delta = max(max_wealth_delta, delta)
                if delta > _R_WEALTH_TOL:
                    bad.append((idx, key, expected_value, replay_value))
            else:
                if not math.isclose(replay_value, expected_value, rel_tol=0, abs_tol=0):
                    bad.append((idx, key, expected_value, replay_value))

    if bad:
        return CheckResult(
            name="replay",
            severity=CheckSeverity.FAIL,
            message=(
                f"{len(bad)} component mismatch(es) across {len(captured_inputs)} steps; "
                f"max r_wealth delta={max_wealth_delta:.3e}"
            ),
            details={"first_mismatches": bad[:5]},
        )
    return CheckResult(
        name="replay",
        severity=CheckSeverity.PASS,
        message=(
            f"all components match across {len(captured_inputs)} steps; "
            f"max r_wealth delta={max_wealth_delta:.3e}"
        ),
    )


__all__ = ["check_reward_replay"]
