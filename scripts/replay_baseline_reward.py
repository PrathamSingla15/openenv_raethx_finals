"""Replay the existing baseline trajectory through the patched reward.

Walks every ``advance_day`` row in ``test/trajectory.jsonl`` from the
recorded baseline run and recomputes the reward using the new
``compute_composite_reward``. Compares against the old recorded values
and reports the cumulative + per-bar mean (= ``score_normalized``).

Verification gate 2 in the R5 plan: the patched reward should put the
broken baseline at ``score_normalized`` in roughly ``[0.50, 0.65]``,
with ``c_alpha`` materially below 0.5 (the agent lost to B&H).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tradebench.rewards.composite import compute_composite_reward  # noqa: E402


DEFAULT_RUN = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "20260425T230125Z__train-test__qwen-qwen3-32b-groq__iter00_baseline"
)


def _load_advance_day_rows(traj_path: Path) -> list[dict]:
    rows: list[dict] = []
    with traj_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("action_type") == "advance_day":
                rows.append(row)
    return rows


def _equal_weight_bench_log_returns(
    initial_value: float, num_bars: int, target_b_and_h_roi: float = 0.1673
) -> list[float]:
    """Synthesize a smooth equal-weight B&H trajectory with the known +16.73% ROI.

    The recorded baseline trajectory does not include benchmark closes, so we
    approximate B&H as a constant-rate log drift such that ``B_T / V_0`` ends
    at the empirical buy-and-hold ROI. This is a faithful proxy for the
    cumulative-alpha computation; the per-bar alphas are roughly constant,
    which slightly suppresses ``c_consistency``'s downside-vol but keeps the
    cumulative ``c_alpha`` honest.
    """

    target_log = math.log(1.0 + target_b_and_h_roi)
    per_bar = target_log / num_bars
    return [per_bar] * num_bars


def replay(run_dir: Path) -> None:
    traj = run_dir / "test" / "trajectory.jsonl"
    if not traj.is_file():
        print(f"missing trajectory: {traj}", file=sys.stderr)
        sys.exit(2)

    rows = _load_advance_day_rows(traj)
    if not rows:
        print("no advance_day rows", file=sys.stderr)
        sys.exit(2)

    initial_value = float(rows[0]["portfolio_value"])
    if rows[0].get("portfolio_value") != rows[0].get("portfolio_value"):
        sys.exit(2)
    initial_value = 100_000.0

    bench_log_per_bar = _equal_weight_bench_log_returns(
        initial_value=initial_value, num_bars=len(rows),
    )

    cum_log_bench = 0.0
    cum_log_agent = 0.0
    high_water = initial_value
    prev_v = initial_value
    recent_alphas: list[float] = []
    new_rewards: list[float] = []
    new_breakdowns: list[dict[str, float]] = []
    old_rewards: list[float] = []

    for i, row in enumerate(rows):
        v_after = float(row["portfolio_value"])
        if v_after <= 0:
            v_after = 1.0

        bar_log_agent = math.log(v_after / prev_v) if prev_v > 0 else 0.0
        bar_log_bench = bench_log_per_bar[i]
        cum_log_agent = math.log(v_after / initial_value)
        cum_log_bench += bar_log_bench
        bar_alpha = bar_log_agent - bar_log_bench
        recent_alphas.append(bar_alpha)
        if len(recent_alphas) > 20:
            recent_alphas.pop(0)

        breakdown = compute_composite_reward(
            value_after=v_after,
            initial_value=initial_value,
            cumulative_log_return=cum_log_agent,
            cumulative_benchmark_log_return=cum_log_bench,
            recent_bar_alphas=tuple(recent_alphas),
            high_watermark=high_water,
            turnover_ratio=0.05,
            hhi=0.20,
            gross_leverage=0.0,
            violations_rules=False,
            violations_hack=False,
        )
        new_rewards.append(breakdown.r_total)
        new_breakdowns.append(breakdown.to_dict())
        old_rewards.append(float(row.get("reward", 0.0)))
        if v_after > high_water:
            high_water = v_after
        prev_v = v_after

    cum_old = sum(old_rewards)
    cum_new = sum(new_rewards)
    score_old_sigmoid = 1.0 / (1.0 + math.exp(-cum_old))
    score_new = cum_new / len(new_rewards)

    print(f"baseline run: {run_dir.name}")
    print(f"bars (advance_day rows): {len(rows)}")
    print(f"initial portfolio: {initial_value:.2f}")
    print(f"final portfolio:   {prev_v:.2f}  (ROI {prev_v/initial_value - 1:+.4%})")
    print()
    print("== old reward (degenerate) ==")
    print(f"  cumulative_reward = {cum_old:+.4f}")
    print(f"  score_normalized  = sigmoid(cum) = {score_old_sigmoid:.4f}")
    print()
    print("== new reward (patched) ==")
    print(f"  cumulative_reward = {cum_new:+.4f}  (in [0, {len(rows)}])")
    print(f"  score_normalized  = mean(per_bar) = {score_new:.4f}")
    print()
    print("per-component cumulative means:")
    for key in (
        "c_alpha",
        "c_return",
        "c_drawdown",
        "c_solvency",
        "c_efficiency",
        "c_diversity",
        "c_consistency",
        "g_compliance",
    ):
        avg = mean(b[key] for b in new_breakdowns)
        print(f"  {key:14s} {avg:.4f}")
    print()

    expected_band = (0.50, 0.65)
    in_band = expected_band[0] <= score_new <= expected_band[1]
    final_alpha = mean(b["c_alpha"] for b in new_breakdowns)
    alpha_below_pivot = final_alpha < 0.50
    print(
        f"GATE 2.a  score_normalized in {expected_band}? {in_band} (got {score_new:.4f})"
    )
    print(
        f"GATE 2.b  mean c_alpha < 0.50 (lost to B&H)? {alpha_below_pivot} (got {final_alpha:.4f})"
    )
    print(
        f"GATE 2.c  all g_compliance == 1.0? "
        f"{all(b['g_compliance'] == 1.0 for b in new_breakdowns)}"
    )

    if not (in_band and alpha_below_pivot):
        sys.exit(1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        type=Path,
        default=DEFAULT_RUN,
        help="Run directory containing test/trajectory.jsonl",
    )
    args = parser.parse_args()
    replay(args.run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
