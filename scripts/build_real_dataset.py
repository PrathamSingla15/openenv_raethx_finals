"""Bake real OHLCV from yfinance into the TradeBench parquet schema.

Pipeline (deterministic, seed-driven):

1. Fetch raw auto-adjusted OHLCV for a pool of ~35 mega-cap US tickers over
   ``[SOURCE_START, SOURCE_END]`` via ``yfinance.download(...)``. Cached
   locally on first run; subsequent runs reuse the cache.

2. For each tier (T1 / T2 / T3):

   - Compute ``manifest_sessions`` = weekday-only dates in the manifest
     window (matches ``sample_dataset._sessions``).
   - Sample ``K`` tickers from the pool without replacement.
   - For each (alias, ticker) pair:
       - Pick a random source window of ``N = len(manifest_sessions)``
         consecutive trading days from that ticker's history.
       - Compute log-returns; add zero-mean Gaussian noise σ=0.0005.
       - Rebuild a close series starting at ``BASE_PRICE = 75.0``.
       - Rebuild OHLV in the same shape as ``_synthetic_bar_series``
         (open ≈ prior_close × jitter, high = max(open,close)×1.005, …).
       - Emit one ``DailyBarRow`` per session and one ``AssetMasterRow``
         per session.

3. Write parquet under ``datasets/real-data/real-v0/`` mirroring the
   ``catalog/sample-v0/`` schema. Drop ``aliases.json`` as a private
   sidecar (audit only — the catalog directory does not include it, so
   ``progressive_fs`` never copies it into the agent sandbox).

Tier corporate-actions are intentionally empty (matches the existing
synthetic tier behavior — auto-adjusted closes already absorb splits and
dividends, so we don't emit split/div rows that would re-double them).

Run::

    uv run --extra realdata python scripts/build_real_dataset.py
    uv run --extra realdata python scripts/build_real_dataset.py --seed 42

Same ``--seed`` produces byte-identical parquet output, which the
verifier's replay-determinism check exercises.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Constants kept in sync with src/tradebench/data/sample_dataset.py.
T1_EPISODE_START = date(2020, 1, 2)
T1_EPISODE_END = date(2020, 3, 31)

T1_ALIASES = tuple(f"tier_t1_a{i:02d}" for i in range(1, 6))

# Shared 10-asset universe for the train/test pair. The same alias maps to
# the same real ticker across both train and test — enforced by the unified
# baker below. Aliases are intentionally NOT prefixed with "_train" / "_test"
# because the agent must treat them as the same tradable instrument across
# the study and rollout phases.
PAIR_ALIASES = tuple(f"tier_a{i:02d}" for i in range(1, 11))

PAIR_TRAIN_BARS = 252
PAIR_TEST_BARS = 120
PAIR_TOTAL_BARS = PAIR_TRAIN_BARS + PAIR_TEST_BARS

# Anchor the train window at the start of 2024; train+test together span
# the next ~372 trading days, so test ends mid-2025. yfinance has continuous
# coverage for every POOL ticker through that window. The 120-bar test is
# long enough to be meaningful given the per-bar record_decision gate
# (~240+ minimum steps), without ballooning wall time.
PAIR_TRAIN_START = date(2024, 1, 2)

POOL: tuple[str, ...] = (
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "V", "JNJ",
    "BRK-B", "UNH", "HD", "MA", "BAC", "KO", "PEP", "WMT", "XOM", "PG",
    "CVX", "ABBV", "LLY", "AVGO", "COST", "MRK", "TMO", "DIS", "ORCL", "NFLX",
    "ADBE", "CRM", "ACN", "CSCO", "TXN",
)

SOURCE_START = date(2018, 1, 1)
SOURCE_END = date(2026, 3, 31)

NOISE_SIGMA = 0.0005
BASE_PRICE = 75.0

TIER_SPECS: tuple[dict, ...] = (
    {"id": "t1", "start": T1_EPISODE_START, "end": T1_EPISODE_END, "aliases": T1_ALIASES},
)


def _weekday_sessions(start: date, end: date) -> list[date]:
    out: list[date] = []
    d = start
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d = date.fromordinal(d.toordinal() + 1)
    return out


def _next_n_weekdays(start: date, n: int) -> list[date]:
    """Return the first ``n`` weekday dates starting at ``start`` (inclusive)."""
    out: list[date] = []
    d = start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d = date.fromordinal(d.toordinal() + 1)
    return out


def _utc_eod(d: date) -> datetime:
    return datetime.combine(d, time(23, 59, 59, 999999), tzinfo=UTC)


def _q(value: float, places: int = 4) -> Decimal:
    return Decimal(str(round(float(value), places)))


def _fetch_pool(cache_path: Path) -> dict[str, pd.DataFrame]:
    """Load auto-adjusted OHLCV for every ticker in POOL.

    Cached per-ticker as parquet under ``cache_path/<TICKER>.parquet`` so
    re-runs are fast and offline-friendly.
    """
    cache_path.mkdir(parents=True, exist_ok=True)
    out: dict[str, pd.DataFrame] = {}

    missing = [t for t in POOL if not (cache_path / f"{t}.parquet").is_file()]
    if missing:
        import yfinance as yf
        print(f"yfinance: fetching {len(missing)} ticker(s): {missing}")
        df = yf.download(
            tickers=missing,
            start=SOURCE_START.isoformat(),
            end=SOURCE_END.isoformat(),
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=True,
        )
        if df is None or df.empty:
            raise RuntimeError("yfinance returned empty dataframe")

        for ticker in missing:
            if len(missing) == 1:
                tdf = df.copy()
            else:
                tdf = df[ticker].copy()  # type: ignore[index]
            tdf = tdf.dropna(how="any")
            tdf.index = pd.to_datetime(tdf.index).tz_localize(None).normalize()
            tdf.to_parquet(cache_path / f"{ticker}.parquet")

    for ticker in POOL:
        out[ticker] = pd.read_parquet(cache_path / f"{ticker}.parquet")
    return out


def _emit_alias_chunk(
    alias: str,
    ticker: str,
    source_window: pd.DataFrame,
    manifest_sessions: list[date],
    rng: random.Random,
    np_rng: np.random.Generator,
) -> tuple[list[dict], list[dict]]:
    """Build (bars_rows, master_rows) for one (alias, source-chunk) pair.

    Computes log-returns, adds Gaussian noise σ = NOISE_SIGMA, cumulates
    from ``log(BASE_PRICE)`` to a new close series, then rebuilds OHLV
    around it. Source window length must equal ``len(manifest_sessions)``.
    """
    n = len(manifest_sessions)
    if len(source_window) != n:
        raise ValueError(
            f"source_window length {len(source_window)} != manifest sessions {n}",
        )

    closes = source_window["Close"].to_numpy(dtype=np.float64)
    volumes = source_window["Volume"].to_numpy(dtype=np.float64)

    log_rets = np.zeros(n, dtype=np.float64)
    log_rets[1:] = np.diff(np.log(np.clip(closes, 1e-9, None)))
    noise = np_rng.normal(0.0, NOISE_SIGMA, size=n)
    noisy_log_rets = log_rets + noise

    log_prices = np.cumsum(noisy_log_rets) + math.log(BASE_PRICE)
    new_closes = np.exp(log_prices)

    bars: list[dict] = []
    master: list[dict] = []

    for i, manifest_date in enumerate(manifest_sessions):
        close_f = float(new_closes[i])
        prior_close = float(new_closes[i - 1]) if i > 0 else close_f
        open_jitter = 1.0 + (rng.random() - 0.5) * 0.004
        open_f = prior_close * open_jitter
        high_f = max(open_f, close_f) * 1.005
        low_f = min(open_f, close_f) * 0.995

        # Rescale volume to the new price level; floor at 100k to keep
        # dollar_volume sensible.
        src_close = float(closes[i]) if closes[i] > 0 else 1.0
        scaled_vol = max(100_000, int(volumes[i] * (src_close / close_f)))

        close_d = _q(close_f)
        open_d = _q(open_f)
        high_d = _q(high_f)
        low_d = _q(low_f)

        bars.append(
            {
                "asset_id": alias,
                "session_date": manifest_date,
                "open": open_d,
                "high": high_d,
                "low": low_d,
                "close": close_d,
                "volume": int(scaled_vol),
                "dollar_volume": close_d * Decimal(scaled_vol),
                "available_at": _utc_eod(manifest_date),
            },
        )
        master.append(
            {
                "asset_id": alias,
                "symbol": alias.upper(),
                "primary_exchange": "XNYS",
                "listing_date": date(2018, 1, 3),
                "delisting_date": None,
                "replaces_asset_id": None,
                "replaced_by_asset_id": None,
                "snapshot_date": manifest_date,
                "available_at": _utc_eod(manifest_date),
            },
        )

    return bars, master


def _build_tier(
    tier_spec: dict,
    pool_data: dict[str, pd.DataFrame],
    rng: random.Random,
    np_rng: np.random.Generator,
) -> tuple[list[dict], list[dict], dict[str, dict]]:
    """For one tier, sample windows + tickers + permutation, emit rows."""

    manifest_sessions = _weekday_sessions(tier_spec["start"], tier_spec["end"])
    n = len(manifest_sessions)
    aliases = tier_spec["aliases"]
    k = len(aliases)

    tickers = rng.sample(list(POOL), k)

    bars: list[dict] = []
    master: list[dict] = []
    tier_meta: dict[str, dict] = {}

    for alias, ticker in zip(aliases, tickers):
        ticker_df = pool_data[ticker]
        if len(ticker_df) < n:
            raise RuntimeError(
                f"{ticker} has {len(ticker_df)} bars; need {n} for tier {tier_spec['id']}",
            )
        max_start = len(ticker_df) - n
        start_idx = rng.randint(0, max_start)
        source_window = ticker_df.iloc[start_idx : start_idx + n]
        src_first = source_window.index[0].date().isoformat()
        src_last = source_window.index[-1].date().isoformat()

        a_bars, a_master = _emit_alias_chunk(
            alias=alias,
            ticker=ticker,
            source_window=source_window,
            manifest_sessions=manifest_sessions,
            rng=rng,
            np_rng=np_rng,
        )
        bars.extend(a_bars)
        master.extend(a_master)

        tier_meta[alias] = {
            "real_ticker": ticker,
            "source_start": src_first,
            "source_end": src_last,
            "manifest_start": manifest_sessions[0].isoformat(),
            "manifest_end": manifest_sessions[-1].isoformat(),
            "n_bars": n,
        }

    return bars, master, tier_meta


def _build_train_test_pair(
    pool_data: dict[str, pd.DataFrame],
    rng: random.Random,
    np_rng: np.random.Generator,
) -> tuple[list[dict], list[dict], dict[str, dict], dict[str, dict], list[date], list[date]]:
    """Bake the unified train+test pair sharing one universe and one calendar.

    Each alias gets ONE source-window of length ``PAIR_TOTAL_BARS`` from its
    real-ticker history. The first ``PAIR_TRAIN_BARS`` source bars become the
    train slice (mapped to ``train_sessions``); the next ``PAIR_TEST_BARS``
    become the test slice (mapped to ``test_sessions``). The same alias is
    backed by the same real ticker across both slices so a strategy derived
    from train is meaningful on test.

    Returns ``(bars, master, train_meta, test_meta, train_sessions, test_sessions)``.
    """

    all_sessions = _next_n_weekdays(PAIR_TRAIN_START, PAIR_TOTAL_BARS)
    train_sessions = all_sessions[:PAIR_TRAIN_BARS]
    test_sessions = all_sessions[PAIR_TRAIN_BARS:]

    tickers = rng.sample(list(POOL), len(PAIR_ALIASES))

    bars: list[dict] = []
    master: list[dict] = []
    train_meta: dict[str, dict] = {}
    test_meta: dict[str, dict] = {}

    for alias, ticker in zip(PAIR_ALIASES, tickers):
        ticker_df = pool_data[ticker]
        if len(ticker_df) < PAIR_TOTAL_BARS:
            raise RuntimeError(
                f"{ticker} has {len(ticker_df)} bars; need {PAIR_TOTAL_BARS} "
                f"for the train+test pair",
            )
        max_start = len(ticker_df) - PAIR_TOTAL_BARS
        start_idx = rng.randint(0, max_start)
        full_window = ticker_df.iloc[start_idx : start_idx + PAIR_TOTAL_BARS]
        train_window = full_window.iloc[:PAIR_TRAIN_BARS]
        test_window = full_window.iloc[PAIR_TRAIN_BARS:]

        train_bars, train_master = _emit_alias_chunk(
            alias=alias,
            ticker=ticker,
            source_window=train_window,
            manifest_sessions=train_sessions,
            rng=rng,
            np_rng=np_rng,
        )
        test_bars, test_master = _emit_alias_chunk(
            alias=alias,
            ticker=ticker,
            source_window=test_window,
            manifest_sessions=test_sessions,
            rng=rng,
            np_rng=np_rng,
        )
        bars.extend(train_bars)
        bars.extend(test_bars)
        master.extend(train_master)
        master.extend(test_master)

        train_meta[alias] = {
            "real_ticker": ticker,
            "source_start": train_window.index[0].date().isoformat(),
            "source_end": train_window.index[-1].date().isoformat(),
            "manifest_start": train_sessions[0].isoformat(),
            "manifest_end": train_sessions[-1].isoformat(),
            "n_bars": PAIR_TRAIN_BARS,
        }
        test_meta[alias] = {
            "real_ticker": ticker,
            "source_start": test_window.index[0].date().isoformat(),
            "source_end": test_window.index[-1].date().isoformat(),
            "manifest_start": test_sessions[0].isoformat(),
            "manifest_end": test_sessions[-1].isoformat(),
            "n_bars": PAIR_TEST_BARS,
        }

    return bars, master, train_meta, test_meta, train_sessions, test_sessions


def _write_parquet(rows: list[dict], path: Path, schema: pa.Schema) -> None:
    """Write rows to parquet with an explicit Arrow schema (preserves Decimal/UTC)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        # Empty file with schema so downstream readers don't choke.
        table = pa.Table.from_pydict({f.name: [] for f in schema}, schema=schema)
    else:
        cols = {f.name: [r[f.name] for r in rows] for f in schema}
        table = pa.Table.from_pydict(cols, schema=schema)
    pq.write_table(table, path)


# Arrow schemas mirror what sample_dataset writes via pandas inference so
# the merged catalog has uniform column types after splice.
_BARS_SCHEMA = pa.schema(
    [
        ("asset_id", pa.string()),
        ("session_date", pa.date32()),
        ("open", pa.decimal128(20, 4)),
        ("high", pa.decimal128(20, 4)),
        ("low", pa.decimal128(20, 4)),
        ("close", pa.decimal128(20, 4)),
        ("volume", pa.int64()),
        ("dollar_volume", pa.decimal128(28, 4)),
        ("available_at", pa.timestamp("us", tz="UTC")),
    ],
)

_MASTER_SCHEMA = pa.schema(
    [
        ("asset_id", pa.string()),
        ("symbol", pa.string()),
        ("primary_exchange", pa.string()),
        ("listing_date", pa.date32()),
        ("delisting_date", pa.date32()),
        ("replaces_asset_id", pa.string()),
        ("replaced_by_asset_id", pa.string()),
        ("snapshot_date", pa.date32()),
        ("available_at", pa.timestamp("us", tz="UTC")),
    ],
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=20260425)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "datasets" / "real-data" / "real-v0",
    )
    ap.add_argument(
        "--cache",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "datasets" / "real-data" / "_yfinance_cache",
    )
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)

    print(f"build_real_dataset: seed={args.seed} out={args.out}")
    print(f"build_real_dataset: fetching/loading source pool ({len(POOL)} tickers)")
    pool_data = _fetch_pool(args.cache)

    all_bars: list[dict] = []
    all_master: list[dict] = []
    aliases_meta: dict = {
        "seed": args.seed,
        "noise_sigma": NOISE_SIGMA,
        "base_price": BASE_PRICE,
        "tiers": {},
    }

    for tier_spec in TIER_SPECS:
        print(
            f"build_real_dataset: tier={tier_spec['id']} "
            f"window={tier_spec['start']}→{tier_spec['end']} "
            f"aliases={len(tier_spec['aliases'])}",
        )
        bars, master, meta = _build_tier(tier_spec, pool_data, rng, np_rng)
        all_bars.extend(bars)
        all_master.extend(master)
        aliases_meta["tiers"][tier_spec["id"]] = meta

    print(
        f"build_real_dataset: train+test pair "
        f"train_bars={PAIR_TRAIN_BARS} test_bars={PAIR_TEST_BARS} "
        f"universe={len(PAIR_ALIASES)} anchor={PAIR_TRAIN_START}",
    )
    pair_bars, pair_master, train_meta, test_meta, train_sessions, test_sessions = (
        _build_train_test_pair(pool_data, rng, np_rng)
    )
    all_bars.extend(pair_bars)
    all_master.extend(pair_master)
    aliases_meta["tiers"]["train"] = train_meta
    aliases_meta["tiers"]["test"] = test_meta
    aliases_meta["pair_calendar"] = {
        "train_start": train_sessions[0].isoformat(),
        "train_end": train_sessions[-1].isoformat(),
        "test_start": test_sessions[0].isoformat(),
        "test_end": test_sessions[-1].isoformat(),
        "universe": list(PAIR_ALIASES),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    bars_path = args.out / "daily_bars" / "part-000.parquet"
    master_path = args.out / "asset_master.parquet"
    actions_path = args.out / "corporate_actions.parquet"
    aliases_path = args.out.parent / "aliases.json"  # sibling of real-v0/, not a child

    _write_parquet(all_bars, bars_path, _BARS_SCHEMA)
    _write_parquet(all_master, master_path, _MASTER_SCHEMA)
    # Empty corp_actions: ``auto_adjust=True`` absorbs splits and dividends.
    _write_parquet(
        [],
        actions_path,
        pa.schema(
            [
                ("asset_id", pa.string()),
                ("action_type", pa.string()),
                ("effective_date", pa.date32()),
                ("ex_date", pa.date32()),
                ("split_from", pa.decimal128(20, 4)),
                ("split_to", pa.decimal128(20, 4)),
                ("dividend_amount", pa.decimal128(20, 4)),
                ("new_symbol", pa.string()),
                ("metadata", pa.string()),
                ("available_at", pa.timestamp("us", tz="UTC")),
            ],
        ),
    )

    aliases_path.write_text(
        json.dumps(aliases_meta, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"build_real_dataset: wrote {len(all_bars)} bars → {bars_path.relative_to(args.out.parent.parent)}")
    print(f"build_real_dataset: wrote {len(all_master)} master rows → {master_path.relative_to(args.out.parent.parent)}")
    print(f"build_real_dataset: aliases sidecar → {aliases_path.relative_to(args.out.parent.parent)}")

    rt_bars = pd.read_parquet(bars_path)
    rt_master = pd.read_parquet(master_path)
    assert len(rt_bars) == len(all_bars), "bars round-trip count mismatch"
    assert len(rt_master) == len(all_master), "master round-trip count mismatch"
    assert rt_bars["close"].notna().all(), "bars contain NaN closes"
    assert (rt_bars["close"].astype(str).map(Decimal) > 0).all(), "negative or zero closes"
    print("build_real_dataset: round-trip validation OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
