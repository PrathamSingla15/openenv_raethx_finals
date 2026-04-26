"""Build the in-context study packet for the train tier.

The study packet is the full train-window OHLCV plus pre-computed summary
statistics, formatted as a compact text block that gets returned in
``TradeObservation.tool_output`` when ``task_tier == "train"``. The agent
reads it, derives a strategy, and emits a single ``record_decision``; no
per-bar rollout, no reward.

Token budget: target < 12K tokens for a 252-bar / 10-asset packet so the
study text fits comfortably alongside the system prompt and leaves room
for the ``record_decision`` response.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]

from tradebench.episodes.models import EpisodeManifest


@dataclass(frozen=True)
class StudyPacket:
    """Renderable study packet for a train-tier episode.

    ``text`` is the human-readable rendering passed to the model.
    ``stats`` is the structured payload echoed in ``tool_metadata``
    (per-asset summary stats keyed by alias) for downstream tooling.
    """

    text: str
    stats: dict[str, dict[str, float]]
    correlations: dict[str, dict[str, float]]
    bars_count: int
    universe: list[str]


def build_study_packet(
    manifest: EpisodeManifest,
    *,
    bars_parquet_path: Path,
    head_tail: int = 5,
) -> StudyPacket:
    """Read the train window from the catalog and render the study packet.

    ``bars_parquet_path`` should point at the merged catalog's
    ``daily_bars/part-000.parquet`` (the same file the agent's sandbox
    sees post-materialization). Filtering to the train window happens
    here so the helper is independent of the runtime's PIT machinery.
    """

    df = pd.read_parquet(bars_parquet_path)
    df = df[df["asset_id"].isin(manifest.universe_asset_ids)].copy()
    start = manifest.episode_start
    end = manifest.episode_end
    df = df[(df["session_date"] >= start) & (df["session_date"] <= end)]
    df["close"] = df["close"].apply(lambda d: float(d) if isinstance(d, Decimal) else float(d))
    df = df.sort_values(["asset_id", "session_date"]).reset_index(drop=True)

    universe = list(manifest.universe_asset_ids)
    closes_wide = df.pivot(index="session_date", columns="asset_id", values="close")
    closes_wide = closes_wide[universe]
    log_returns = (closes_wide.apply(lambda s: s.map(math.log)).diff()).iloc[1:]

    stats = _per_asset_stats(closes_wide, log_returns)
    corr = _correlation_matrix(log_returns)

    text = _render_text(
        manifest=manifest,
        universe=universe,
        closes_wide=closes_wide,
        stats=stats,
        corr=corr,
        head_tail=head_tail,
    )
    return StudyPacket(
        text=text,
        stats=stats,
        correlations=corr,
        bars_count=int(closes_wide.shape[0]),
        universe=universe,
    )


def _per_asset_stats(
    closes_wide: pd.DataFrame,
    log_returns: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    n = len(log_returns)
    for alias in closes_wide.columns:
        rets = log_returns[alias].dropna().to_numpy()
        if len(rets) == 0:
            continue
        mean = float(rets.mean())
        vol = float(rets.std(ddof=1)) if len(rets) > 1 else 0.0
        sharpe = (mean / vol * math.sqrt(252)) if vol > 0 else 0.0
        cum_log = rets.cumsum()
        peak = -float("inf")
        max_dd = 0.0
        for x in cum_log:
            peak = max(peak, x)
            dd = float(math.exp(peak - x) - 1.0) * -1.0
            max_dd = min(max_dd, dd)
        first_close = float(closes_wide[alias].iloc[0])
        last_close = float(closes_wide[alias].iloc[-1])
        total_ret = (last_close / first_close) - 1.0 if first_close > 0 else 0.0
        out[alias] = {
            "mean_log_return": mean,
            "vol_daily": vol,
            "sharpe_annualized": sharpe,
            "max_drawdown": max_dd,
            "total_return": total_ret,
            "best_day": float(rets.max()),
            "worst_day": float(rets.min()),
            "first_close": first_close,
            "last_close": last_close,
            "n_bars": int(n) + 1,
        }
    return out


def _correlation_matrix(log_returns: pd.DataFrame) -> dict[str, dict[str, float]]:
    corr = log_returns.corr()
    out: dict[str, dict[str, float]] = {}
    for alias in corr.index:
        row: dict[str, float] = {}
        for other in corr.columns:
            value = corr.loc[alias, other]
            row[str(other)] = float(value) if pd.notna(value) else 0.0
        out[str(alias)] = row
    return out


def _render_text(
    *,
    manifest: EpisodeManifest,
    universe: list[str],
    closes_wide: pd.DataFrame,
    stats: dict[str, dict[str, float]],
    corr: dict[str, dict[str, float]],
    head_tail: int,
) -> str:
    n_bars = closes_wide.shape[0]
    lines: list[str] = []
    lines.append(
        f"TRAIN STUDY PACKET ({n_bars} trading days, {len(universe)} assets, fully revealed)",
    )
    lines.append(
        f"Calendar: {manifest.episode_start.isoformat()} -> {manifest.episode_end.isoformat()}",
    )
    lines.append(
        "Note: aliases below are stable across train and test. The same alias "
        "in the test rollout refers to the same underlying instrument.",
    )
    lines.append("")
    lines.append("Per-asset summary (daily log returns over the train window):")
    lines.append(
        "  alias    | mean_ret  | vol_d  | sharpe | max_dd  | total_ret | best_day  | worst_day"
    )
    lines.append(
        "  ---------+-----------+--------+--------+---------+-----------+-----------+----------"
    )
    for alias in universe:
        s = stats.get(alias)
        if s is None:
            continue
        lines.append(
            f"  {alias} | {s['mean_log_return']:+.5f} | {s['vol_daily']:.4f} "
            f"| {s['sharpe_annualized']:+.2f}  | {s['max_drawdown']:+.3f}  "
            f"| {s['total_return']:+.4f}  | {s['best_day']:+.4f}  | {s['worst_day']:+.4f}",
        )
    lines.append("")
    lines.append(f"{len(universe)}x{len(universe)} correlation matrix (daily log returns):")
    header = "         " + "  ".join(a.replace("tier_", "") for a in universe)
    lines.append(header)
    for alias in universe:
        row = corr.get(alias, {})
        cells = "  ".join(f"{row.get(other, 0.0):+0.2f}" for other in universe)
        lines.append(f"  {alias.replace('tier_', '')}  {cells}")
    lines.append("")
    lines.append(f"Daily close prices (head {head_tail} / tail {head_tail}):")
    head = closes_wide.head(head_tail)
    tail = closes_wide.tail(head_tail)
    lines.append(_format_close_table(head, universe))
    lines.append("  ...")
    lines.append(_format_close_table(tail, universe))
    lines.append("")
    lines.append(
        "Full series available via sandbox_exec at "
        "WORKSPACE_DATA/daily_bars/part-000.parquet "
        "(filter for session_date <= train.episode_end).",
    )
    return "\n".join(lines)


def _format_close_table(df: pd.DataFrame, universe: list[str]) -> str:
    header_aliases = "  ".join(f"{a.replace('tier_', ''):>7s}" for a in universe)
    lines = [f"  {'date':<10s}  {header_aliases}"]
    for session_date, row in df.iterrows():
        cells = "  ".join(f"{float(row[a]):7.2f}" for a in universe)
        lines.append(f"  {_iso(session_date)}  {cells}")
    return "\n".join(lines)


def _iso(value: object) -> str:
    if isinstance(value, date):
        return value.isoformat()
    if hasattr(value, "date"):
        return value.date().isoformat()  # type: ignore[no-any-return]
    return str(value)


__all__ = ["StudyPacket", "build_study_packet"]
