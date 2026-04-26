"""Verifier orchestrator + CLI.

``run_verifier(tier, seed)`` builds an in-process ``TradeBenchEnvironment``,
drives it through the tier with a deterministic cash strategy (no orders,
just record_decision → advance_day cycles), captures the trajectory, and
runs each named check on it. Returns a :class:`VerifierReport`.

``main()`` is the argparse entry point. ``__main__.py`` calls this so
``python -m tradebench.verifier --tier t1 --seed 42`` works out of the box.
"""

from __future__ import annotations

import argparse
import sys
from decimal import Decimal
from typing import Any

try:
    from ...models import TradeAction, TradeObservation  # type: ignore[no-redef]
except (ImportError, ValueError):  # pragma: no cover
    from models import TradeAction, TradeObservation  # type: ignore[no-redef]

from ._types import CheckSeverity, VerifierReport
from .conformance import check_action_rejection, check_observation_schemas
from .determinism import check_determinism
from .leak import check_filesystem_audit, check_no_future_fit
from .replay import check_reward_replay


def _build_env() -> Any:
    # Imported here so the CLI doesn't pull the FastAPI/server stack at
    # import time.
    try:
        from server.tradebench_environment import TradeBenchEnvironment
    except ImportError:  # pragma: no cover - layout fallback
        from server.tradebench_environment import TradeBenchEnvironment  # type: ignore[no-redef]
    return TradeBenchEnvironment()


def _no_op_decision_payload() -> dict[str, Any]:
    return {
        "regime_label": "verifier",
        "edge_summary": "no-op decision for verifier sweep",
        "intended_exposure": "0.0",
        "top_convictions": [],
        "uncertainty": "low",
    }


def _drive_episode(
    env: Any,
    *,
    tier: str,
    seed: int,
    max_bars: int,
) -> dict[str, Any]:
    """Drive one cash-only episode; return per-bar trajectory + summary.

    The trajectory captures every observation (post-step). This keeps the
    leak / conformance checks independent of the strategy: any broken
    schema or future partition shows up regardless of orders placed.
    """

    obs_after_reset = env.reset(task_tier=tier, seed=seed)
    trajectory: list[TradeObservation] = []
    action_log: list[str] = []

    bars_advanced = 0
    obs = obs_after_reset
    while not obs.done and bars_advanced < max_bars:
        decision = env.step(
            TradeAction(
                action_type="record_decision",
                payload=_no_op_decision_payload(),
            ),
        )
        action_log.append("record_decision")
        trajectory.append(decision)

        advance = env.step(TradeAction(action_type="advance_day", payload={}))
        action_log.append("advance_day")
        trajectory.append(advance)
        bars_advanced += 1
        obs = advance

    final_value = float(obs.portfolio_value)
    runtime = env._runtime  # noqa: SLF001 - verifier scope intentionally white-box
    summary = {
        "final_value": final_value,
        "action_log": tuple(action_log),
        "event_count": len(runtime.ledger_events()) if runtime is not None else 0,
        "trajectory": trajectory,
        "current_date": runtime.current_date if runtime is not None else None,
        "workspace_data_dir": runtime.workspace.host_data_dir
        if runtime is not None
        else None,
        "reward_inputs_log": list(runtime._reward_inputs_log)  # noqa: SLF001
        if runtime is not None
        else [],
        "reward_breakdown_log": list(runtime._reward_breakdown_log)  # noqa: SLF001
        if runtime is not None
        else [],
    }
    return summary


def run_verifier(
    *,
    tier: str = "t1",
    seed: int = 42,
    max_bars: int = 10,
) -> VerifierReport:
    """Run every verifier check on one tier; return the aggregate report.

    ``max_bars`` caps the cash-only sweep so the verifier completes fast
    even on T3 (default 10 bars is enough to exercise every code path).
    Use a larger cap for end-to-end audit runs.
    """

    report = VerifierReport(tier=tier, seed=seed)

    env = _build_env()
    summary = _drive_episode(env, tier=tier, seed=seed, max_bars=max_bars)

    report.append(check_observation_schemas(summary["trajectory"]))

    runtime = env._runtime  # noqa: SLF001
    actions = list(summary["trajectory"][-1].available_actions or []) if summary["trajectory"] else []
    report.append(check_action_rejection(env, current_actions=actions))

    if summary["workspace_data_dir"] is not None and summary["current_date"] is not None:
        report.append(
            check_filesystem_audit(
                sandbox_data_dir=summary["workspace_data_dir"],
                current_date=summary["current_date"],
                step_index=len(summary["trajectory"]),
            ),
        )
    if runtime is not None and summary["current_date"] is not None:
        report.append(check_no_future_fit(runtime, summary["current_date"]))

    if summary["reward_inputs_log"]:
        report.append(
            check_reward_replay(
                captured_inputs=summary["reward_inputs_log"],
                captured_breakdowns=summary["reward_breakdown_log"],
            ),
        )

    # Re-run from scratch with the same seed; rebuild the env each call so
    # no shared state leaks between the two replays.
    def _replay(s: int) -> dict[str, Any]:
        local_env = _build_env()
        local_summary = _drive_episode(
            local_env,
            tier=tier,
            seed=s,
            max_bars=max_bars,
        )
        return {
            "final_value": local_summary["final_value"],
            "action_log": local_summary["action_log"],
            "event_count": local_summary["event_count"],
        }

    report.append(check_determinism(run_episode=_replay, seed=seed))

    return report


def format_report(report: VerifierReport) -> str:
    """Render the report in the CLI-style PASS/FAIL block."""

    lines = [f"TradeBench verifier — tier={report.tier} seed={report.seed}"]
    width = max((len(r.name) for r in report.results), default=0)
    for r in report.results:
        marker = {"pass": "PASS", "warn": "WARN", "fail": "FAIL"}[r.severity.value]
        lines.append(f"  {marker:<4}  {r.name:<{width}}  {r.message}")
    if report.has_errors:
        lines.append("RESULT: FAIL")
    elif report.has_warnings:
        lines.append("RESULT: PASS (with warnings)")
    else:
        lines.append("RESULT: PASS")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tradebench.verifier",
        description="Assert TradeBench env conformance, leak-freedom, replay, determinism.",
    )
    parser.add_argument("--tier", default="t1", choices=("t1", "train", "test"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-bars",
        type=int,
        default=10,
        help="Max bars to advance during the verifier sweep (default 10).",
    )
    args = parser.parse_args(argv)

    report = run_verifier(tier=args.tier, seed=args.seed, max_bars=args.max_bars)
    print(format_report(report), flush=True)
    return 1 if report.has_errors else 0


__all__ = ["format_report", "main", "run_verifier"]
