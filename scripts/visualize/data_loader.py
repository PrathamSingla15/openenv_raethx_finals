"""Shared loader for reflection artifacts. GLM-aware: rows with no summary.json
are tagged status="in_flight" so viz scripts can plot dashed/hatched segments."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
DATASETS_ROOT = PROJECT_ROOT / "datasets"

# Per-model iter mapping. iter 0 = baseline; iters 1..5 = reflection rollouts.
# Each tuple: (iter_number, run_dir_name, label).
MODEL_RUNS: dict[str, list[tuple[int, str, str]]] = {
    "qwen-qwen3-32b-groq": [
        (0, "20260426T014336Z__train-test__qwen-qwen3-32b-groq", "baseline"),
        (1, "20260426T022442Z__train-test__qwen-qwen3-32b-groq__iter01_reflect", "iter1"),
        (2, "20260426T025319Z__train-test__qwen-qwen3-32b-groq__iter02_reflect", "iter2"),
        (3, "20260426T032611Z__train-test__qwen-qwen3-32b-groq__iter01_reflect", "iter3"),
        (4, "20260426T035335Z__train-test__qwen-qwen3-32b-groq__iter02_reflect", "iter4"),
        (5, "20260426T042137Z__train-test__qwen-qwen3-32b-groq__iter03_reflect", "iter5"),
    ],
    "zai-glm-5.1-together": [
        (0, "20260426T035937Z__train-test__zai-org-glm-5-1-together", "baseline"),
        (1, "20260426T044349Z__train-test__zai-org-glm-5-1-together__iter01_reflect", "iter1"),
        # iters 2..5 appended once GLM reflection completes — see Phase 8 in plan.
    ],
}

MODEL_LABELS: dict[str, str] = {
    "qwen-qwen3-32b-groq": "Qwen3-32B (Groq)",
    "zai-glm-5.1-together": "GLM-5.1 (Together)",
}


@dataclass
class IterRow:
    iter: int
    label: str
    run_dir: Path
    status: str  # "complete" | "in_flight"
    score_normalized: float | None = None
    final_portfolio_value: float | None = None
    initial_portfolio_value: float | None = None
    roi_pct: float | None = None
    bars_completed: int | None = None
    cumulative_reward: float | None = None
    final_score: float | None = None


def load_model_iters(model_id: str) -> list[IterRow]:
    rows: list[IterRow] = []
    for it, run_name, label in MODEL_RUNS[model_id]:
        run_dir = ARTIFACTS_ROOT / "runs" / run_name
        summary = run_dir / "test" / "summary.json"
        if not summary.is_file():
            rows.append(IterRow(it, label, run_dir, status="in_flight"))
            continue
        s = json.loads(summary.read_text())
        rows.append(
            IterRow(
                iter=it,
                label=label,
                run_dir=run_dir,
                status="complete",
                score_normalized=float(s["score_normalized"]),
                final_portfolio_value=float(s["final_portfolio_value"]),
                initial_portfolio_value=float(s["initial_portfolio_value"]),
                roi_pct=(float(s["final_portfolio_value"]) / float(s["initial_portfolio_value"]) - 1.0) * 100.0,
                bars_completed=int(s["bars_completed"]),
                cumulative_reward=float(s["cumulative_reward"]),
                final_score=float(s["final_score"]),
            )
        )
    return rows


_GATE_REJECT_PREFIX = "Long-horizon planning gate:"


def load_advance_day_rows(run_dir: Path) -> list[dict]:
    """Return all SUCCESSFUL advance_day events from test/trajectory.jsonl.

    Skips per-bar gate rejections (the env returns the row but doesn't tick
    the clock; tool_output_excerpt starts with the gate message).
    """
    traj = run_dir / "test" / "trajectory.jsonl"
    rows: list[dict] = []
    with traj.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("action_type") != "advance_day":
                continue
            if row.get("error"):
                continue
            excerpt = row.get("tool_output_excerpt") or ""
            if excerpt.startswith(_GATE_REJECT_PREFIX):
                continue
            rows.append(row)
    return rows


def load_action_counts(run_dir: Path) -> Counter:
    """Count test-phase actions from trajectory.jsonl.

    Filters out:
    - rows with explicit error field set
    - advance_day calls that hit the per-bar record_decision gate (these
      did not advance state and shouldn't be counted as "successful actions").
    """
    traj = run_dir / "test" / "trajectory.jsonl"
    counts: Counter = Counter()
    if not traj.is_file():
        return counts
    with traj.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("error"):
                continue
            atype = row.get("action_type") or "unknown"
            if atype == "advance_day":
                excerpt = row.get("tool_output_excerpt") or ""
                if excerpt.startswith(_GATE_REJECT_PREFIX):
                    continue
            counts[atype] += 1
    return counts


def load_prompt(run_dir: Path) -> str:
    """Read the system prompt that drove a given run."""
    return (run_dir / "system_prompt.txt").read_text(encoding="utf-8")


def episode_manifest() -> dict:
    return json.loads(
        (DATASETS_ROOT / "catalog" / "sample-v2" / "episode_manifests" / "tier_test.json").read_text()
    )


def equal_weight_bnh_series(dates: Iterable[str], initial_value: float = 100_000.0) -> list[float]:
    """Compute equal-weight B&H portfolio value at each agent-date.

    Reads the daily_bars parquet, takes universe close marks at each requested
    date, and returns the B&H portfolio value (set equal weights at the first
    date, no rebalancing thereafter)."""
    import pandas as pd

    manifest = episode_manifest()
    universe = list(manifest["universe_asset_ids"])
    bars_path = DATASETS_ROOT / "catalog" / "sample-v2" / "daily_bars" / "part-000.parquet"
    df = pd.read_parquet(bars_path, columns=["asset_id", "session_date", "close"])
    df = df[df["asset_id"].isin(universe)].copy()
    df["session_date"] = pd.to_datetime(df["session_date"]).dt.strftime("%Y-%m-%d")
    df["close"] = df["close"].astype(float)  # close is stored as Decimal in the parquet

    pivot = df.pivot(index="session_date", columns="asset_id", values="close").sort_index()
    dates = list(dates)
    pivot = pivot.reindex(dates).ffill()  # forward-fill in case a name has no print on a holiday

    # Equal weight at the first agent-date (= bar 0).
    first_close = pivot.iloc[0]
    qty = (initial_value / len(universe)) / first_close  # asset_id -> shares
    bnh_values = (pivot * qty).sum(axis=1).tolist()
    return bnh_values


def replay_reward_components(run_dir: Path) -> list[dict]:
    """Recompute the new convex composite reward + per-component breakdown
    for every advance_day in a run. Uses the live ``compute_composite_reward``
    so drift in the reward module is reflected.
    """
    import sys

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    if str(PROJECT_ROOT / "src") not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from tradebench.rewards.composite import compute_composite_reward  # type: ignore

    rows = load_advance_day_rows(run_dir)
    if not rows:
        return []
    manifest = episode_manifest()
    initial_value = float(manifest.get("initial_cash", 100_000))

    dates = [r["current_date"] for r in rows]
    bnh_values = equal_weight_bnh_series(dates, initial_value=initial_value)

    breakdowns: list[dict] = []
    high_water = initial_value
    prev_v = initial_value
    cum_log_bench = 0.0
    recent_alphas: list[float] = []
    prev_bnh = bnh_values[0]

    for i, row in enumerate(rows):
        v_after = float(row["portfolio_value"])
        if v_after <= 0:
            v_after = 1.0
        bnh_t = bnh_values[i]
        bar_log_agent = math.log(v_after / prev_v) if prev_v > 0 else 0.0
        bar_log_bench = math.log(bnh_t / prev_bnh) if prev_bnh > 0 else 0.0
        cum_log_agent = math.log(v_after / initial_value)
        cum_log_bench = math.log(bnh_t / initial_value)

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
            turnover_ratio=0.05,  # approximation; trajectory doesn't capture this directly
            hhi=0.20,
            gross_leverage=0.0,
            violations_rules=False,
            violations_hack=False,
        )
        d = breakdown.to_dict()
        d["bar"] = i
        d["date"] = row["current_date"]
        d["portfolio_value"] = v_after
        d["bnh_value"] = bnh_t
        d["bar_alpha"] = bar_alpha
        d["cum_alpha"] = cum_log_agent - cum_log_bench
        breakdowns.append(d)

        if v_after > high_water:
            high_water = v_after
        prev_v = v_after
        prev_bnh = bnh_t

    return breakdowns


__all__ = [
    "ARTIFACTS_ROOT",
    "DATASETS_ROOT",
    "MODEL_RUNS",
    "MODEL_LABELS",
    "PROJECT_ROOT",
    "IterRow",
    "load_model_iters",
    "load_advance_day_rows",
    "load_action_counts",
    "load_prompt",
    "episode_manifest",
    "equal_weight_bnh_series",
    "replay_reward_components",
]
