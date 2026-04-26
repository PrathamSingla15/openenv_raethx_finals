"""Deterministic tiny catalog used by unit tests (splits, dividends, rename, delist).

Also seats the three production tiers used by the hackathon driver:

- T1 (debug): 5 assets, 60 bars (2020-01-02 to 2020-03-31). Smoke-test tier.
- TRAIN (study packet): 10 shared aliases, 252 bars from 2024-01-02. Fully
  revealed in-context at episode reset; no per-bar rollout.
- TEST (held-out rollout): the same 10 aliases, 120 bars starting the
  trading day after TRAIN ends. Walked one bar at a time; reward is what
  we score.

Tier bars come from real OHLCV baked by ``scripts/build_real_dataset.py``;
the fixture rows below stay GBM-generated so ``tests/`` (corp-actions,
late-bar PIT, rename/delist) remain green.
"""

from __future__ import annotations

import json
import math
import random
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]

from tradebench.data.models import utc_end_of_day
from tradebench.episodes.models import EpisodeManifest, SplitSpec

SAMPLE_DATASET_VERSION = "sample-v2"

# Where the baked real-data parquet (produced by ``scripts/build_real_dataset.py``)
# lives. Tier asset bars + master rows are spliced in from here; the fixture
# assets (``tb_sample_*``) stay GBM-generated below. The aliases.json sidecar
# is a SIBLING of this directory (``datasets/real-data/aliases.json``) — it is
# never copied into the agent sandbox because ``progressive_fs`` walks the
# catalog version dir, not the real-data dir.
REAL_DATA_VERSION = "real-v0"
REAL_DATA_SUBDIR = Path("real-data") / REAL_DATA_VERSION

ASSET_LIQUID = "tb_sample_liquid"
ASSET_DEAD = "tb_sample_dead"
ASSET_RENAME_OLD = "tb_sample_rename_old"
ASSET_RENAME_NEW = "tb_sample_rename_new"

SYMBOL_LIQUID = "LIQUID"
SYMBOL_DEAD = "DEADCO"
SYMBOL_RENAME_OLD = "OLDY"
SYMBOL_RENAME_NEW = "NEWY"

SESSION_START = date(2020, 1, 2)
SESSION_END = date(2020, 3, 31)

SPLIT_DATE = date(2020, 2, 3)
DIVIDEND_DATE = date(2020, 3, 2)
RENAME_DATE = date(2020, 2, 15)
DELIST_DATE = date(2020, 3, 16)

LATE_BAR_SESSION = date(2020, 1, 15)
LATE_BAR_AVAILABLE = date(2020, 1, 22)

DIVIDEND_LATE_AVAILABLE_SESSION = date(2020, 3, 10)
DIVIDEND_LATE_AVAILABLE_AT = date(2020, 3, 18)

CALENDAR_START = date(2020, 1, 2)
CALENDAR_END = date(2025, 6, 30)

T1_EPISODE_START = date(2020, 1, 2)
T1_EPISODE_END = date(2020, 3, 31)

# Train/test pair calendar (shared 10-asset universe, contiguous trading
# days). The exact dates below are weekday-counted from PAIR_TRAIN_START
# in scripts/build_real_dataset.py: 252 train bars + 120 test bars.
TRAIN_EPISODE_START = date(2024, 1, 2)
TRAIN_EPISODE_END = date(2024, 12, 18)
TEST_EPISODE_START = date(2024, 12, 19)
TEST_EPISODE_END = date(2025, 6, 4)

T1_ASSET_IDS: tuple[str, ...] = tuple(f"tier_t1_a{i:02d}" for i in range(1, 6))

# 10-asset shared universe — train and test reference exactly these aliases,
# in the same order the baker writes them.
PAIR_ASSET_IDS: tuple[str, ...] = tuple(f"tier_a{i:02d}" for i in range(1, 11))

ALL_TIER_ASSETS: tuple[str, ...] = (*T1_ASSET_IDS, *PAIR_ASSET_IDS)


def _sessions(start: date = SESSION_START, end: date = SESSION_END) -> list[date]:
    days: list[date] = []
    current = start
    while current <= end:
        if current.weekday() < 5:
            days.append(current)
        current = date.fromordinal(current.toordinal() + 1)
    return days


def _eod(d: date) -> datetime:
    return utc_end_of_day(d)


def _resolve_real_data_dir(dataset_root: Path) -> Path | None:
    """Locate the baked real-data parquet directory.

    Search order:
      1. ``<dataset_root>/real-data/real-v0/`` — for Docker (dataset_root is
         the canonical ``/app/datasets``) and local dev (``./datasets``).
      2. ``<repo-root>/datasets/real-data/real-v0/`` — fallback for pytest,
         which uses a ``tmp_path`` dataset_root that doesn't contain the
         baked parquet. Anchored to ``__file__`` so it works in source
         layout; in a wheel install this fallback is a no-op (parents[4]
         resolves to site-packages root, no real-data there).

    Returns the first directory that contains ``daily_bars/part-000.parquet``
    AND ``asset_master.parquet``, or ``None`` if neither location has them.
    """

    candidates = [
        (dataset_root / REAL_DATA_SUBDIR).expanduser().resolve(),
        (Path(__file__).resolve().parents[3] / "datasets" / REAL_DATA_SUBDIR).resolve(),
    ]
    for candidate in candidates:
        bars = candidate / "daily_bars" / "part-000.parquet"
        master = candidate / "asset_master.parquet"
        if bars.is_file() and master.is_file():
            return candidate
    return None


def _load_real_tier_data(
    dataset_root: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Load baked real-OHLCV bars + master rows for tier assets.

    Returns ``(bars_rows, master_rows)`` shaped identically to what the
    GBM generator + ``_master_row`` produced for tier assets, so the
    spliced parquet ends up byte-equivalent in schema to the v0 layout.

    Raises ``FileNotFoundError`` with a pointer to the bake script if the
    parquet has not been produced yet — keeps "missing data" loud rather
    than silently degrading to GBM.
    """

    real_dir = _resolve_real_data_dir(dataset_root)
    if real_dir is None:
        msg = (
            "Baked real-data parquet not found in dataset_root or repo. "
            "Run `uv run --extra realdata python scripts/build_real_dataset.py` "
            "to produce it before regenerating the sample catalog."
        )
        raise FileNotFoundError(msg)

    bars_df = pd.read_parquet(real_dir / "daily_bars" / "part-000.parquet")
    master_df = pd.read_parquet(real_dir / "asset_master.parquet")
    return (
        bars_df.to_dict("records"),  # type: ignore[no-any-return]
        master_df.to_dict("records"),  # type: ignore[no-any-return]
    )


def ensure_sample_dataset(dataset_root: Path) -> Path:
    """Ensure the deterministic sample dataset exists under ``dataset_root``.

    The check uses the *latest* expected artifact (the T3 tier manifest) so
    older deployments that only have ``sample_ep_001.json`` are detected as
    out-of-date and regenerated.
    """

    root = dataset_root.expanduser().resolve()
    sentinel = (
        root
        / "catalog"
        / SAMPLE_DATASET_VERSION
        / "episode_manifests"
        / "tier_test.json"
    )
    if sentinel.is_file():
        return root

    root.mkdir(parents=True, exist_ok=True)
    write_sample_dataset(root)
    return root


def write_sample_dataset(dataset_root: Path) -> Path:
    """
    Write ``catalog/<SAMPLE_DATASET_VERSION>/`` under ``dataset_root``.

    Returns the version directory path.

    The catalog covers the existing fixture window (Q1 2020) plus an
    extended calendar through 2024-12-31 to seat the T1/T2/T3 tier
    datasets. Tier bars and master rows are appended to the same parquet
    files; existing fixture rows are untouched so corp-action / PIT-
    visibility tests still pass.
    """

    root = dataset_root.expanduser().resolve()
    version_dir = root / "catalog" / SAMPLE_DATASET_VERSION
    bars_dir = version_dir / "daily_bars"
    manifests_dir = version_dir / "episode_manifests"
    bars_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    fixture_sessions = _sessions(SESSION_START, SESSION_END)
    extended_sessions = _sessions(CALENDAR_START, CALENDAR_END)

    calendar_df = pd.DataFrame({"session_date": extended_sessions})
    calendar_df.to_parquet(version_dir / "calendar.parquet", index=False)

    bars: list[dict[str, object]] = []
    for s in fixture_sessions:
        if s < DELIST_DATE:
            bars.extend(_bar_rows(ASSET_DEAD, s, close=Decimal("5.00")))
        bars.extend(_bar_rows(ASSET_LIQUID, s, close=Decimal("100.00")))
        if s < RENAME_DATE:
            bars.extend(_bar_rows(ASSET_RENAME_OLD, s, close=Decimal("12.00")))
        if s >= RENAME_DATE:
            bars.extend(_bar_rows(ASSET_RENAME_NEW, s, close=Decimal("12.50")))

    late = next(
        b
        for b in bars
        if b["asset_id"] == ASSET_LIQUID and b["session_date"] == LATE_BAR_SESSION
    )
    late["available_at"] = _eod(LATE_BAR_AVAILABLE)

    # Tier bars come from baked yfinance OHLCV. Tier corp-actions are
    # intentionally empty because ``yf.download(auto_adjust=True)`` already
    # absorbs splits and dividends into the close price.
    real_tier_bars, real_tier_master = _load_real_tier_data(root)
    bars.extend(real_tier_bars)

    bars_df = pd.DataFrame(bars)
    bars_df.to_parquet(bars_dir / "part-000.parquet", index=False)

    masters: list[dict[str, object]] = []
    for s in fixture_sessions:
        masters.append(
            _master_row(
                ASSET_LIQUID,
                SYMBOL_LIQUID,
                s,
                delisting_date=None,
                replaces=None,
                replaced_by=None,
            ),
        )
        if s < DELIST_DATE:
            masters.append(
                _master_row(
                    ASSET_DEAD,
                    SYMBOL_DEAD,
                    s,
                    delisting_date=None,
                    replaces=None,
                    replaced_by=None,
                ),
            )
        else:
            masters.append(
                _master_row(
                    ASSET_DEAD,
                    SYMBOL_DEAD,
                    s,
                    delisting_date=DELIST_DATE,
                    replaces=None,
                    replaced_by=None,
                ),
            )
        if s < RENAME_DATE:
            masters.append(
                _master_row(
                    ASSET_RENAME_OLD,
                    SYMBOL_RENAME_OLD,
                    s,
                    delisting_date=None,
                    replaces=None,
                    replaced_by=ASSET_RENAME_NEW,
                ),
            )
        if s >= RENAME_DATE:
            masters.append(
                _master_row(
                    ASSET_RENAME_NEW,
                    SYMBOL_RENAME_NEW,
                    s,
                    delisting_date=None,
                    replaces=ASSET_RENAME_OLD,
                    replaced_by=None,
                ),
            )

    masters.extend(real_tier_master)

    master_df = pd.DataFrame(masters)
    master_df.to_parquet(version_dir / "asset_master.parquet", index=False)

    actions: list[dict[str, object]] = [
        {
            "asset_id": ASSET_LIQUID,
            "action_type": "split",
            "effective_date": SPLIT_DATE,
            "ex_date": SPLIT_DATE,
            "split_from": Decimal("1"),
            "split_to": Decimal("2"),
            "dividend_amount": None,
            "new_symbol": None,
            "metadata": json.dumps({}),
            "available_at": _eod(SPLIT_DATE),
        },
        {
            "asset_id": ASSET_LIQUID,
            "action_type": "cash_dividend",
            "effective_date": DIVIDEND_DATE,
            "ex_date": DIVIDEND_DATE,
            "split_from": None,
            "split_to": None,
            "dividend_amount": Decimal("0.42"),
            "new_symbol": None,
            "metadata": json.dumps({}),
            "available_at": _eod(DIVIDEND_DATE),
        },
        {
            "asset_id": ASSET_RENAME_OLD,
            "action_type": "ticker_change",
            "effective_date": RENAME_DATE,
            "ex_date": RENAME_DATE,
            "split_from": None,
            "split_to": None,
            "dividend_amount": None,
            "new_symbol": SYMBOL_RENAME_NEW,
            "metadata": json.dumps({}),
            "available_at": _eod(RENAME_DATE),
        },
        {
            "asset_id": ASSET_DEAD,
            "action_type": "delisting",
            "effective_date": DELIST_DATE,
            "ex_date": DELIST_DATE,
            "split_from": None,
            "split_to": None,
            "dividend_amount": None,
            "new_symbol": None,
            "metadata": json.dumps({}),
            "available_at": _eod(DELIST_DATE),
        },
        {
            "asset_id": ASSET_LIQUID,
            "action_type": "cash_dividend",
            "effective_date": DIVIDEND_LATE_AVAILABLE_SESSION,
            "ex_date": DIVIDEND_LATE_AVAILABLE_SESSION,
            "split_from": None,
            "split_to": None,
            "dividend_amount": Decimal("0.10"),
            "new_symbol": None,
            "metadata": json.dumps({}),
            "available_at": _eod(DIVIDEND_LATE_AVAILABLE_AT),
        },
    ]
    actions_df = pd.DataFrame(actions)
    actions_df.to_parquet(version_dir / "corporate_actions.parquet", index=False)

    fundamentals = pd.DataFrame(
        [
            {
                "asset_id": ASSET_LIQUID,
                "fiscal_period_start": date(2019, 1, 1),
                "fiscal_period_end": date(2019, 12, 31),
                "revenue": Decimal("1_000_000"),
                "net_income": Decimal("100_000"),
                "shares_outstanding": Decimal("1_000_000"),
                "available_at": datetime(2020, 2, 15, 16, 0, 0, tzinfo=UTC),
            },
        ],
    )
    fundamentals.to_parquet(version_dir / "fundamentals_pti.parquet", index=False)

    manifest = EpisodeManifest(
        dataset_version=SAMPLE_DATASET_VERSION,
        task_id="sample_ep_001",
        split=SplitSpec(name="train"),
        warmup_start=date(2019, 6, 3),
        episode_start=SESSION_START,
        episode_end=SESSION_END,
        initial_cash=Decimal("100000"),
        universe_asset_ids=[
            ASSET_LIQUID,
            ASSET_DEAD,
            ASSET_RENAME_OLD,
            ASSET_RENAME_NEW,
        ],
        cost_model_version="v0",
        scorer_version="v0",
        sandbox_image_digest="sha256:sample",
        dependency_lock_digest="sha256:lock",
        regime_labels={"sample": "synthetic"},
    )
    (manifests_dir / "sample_ep_001.json").write_text(
        manifest.model_dump_json(indent=2),
        encoding="utf-8",
    )

    for tier_spec in _TIER_MANIFEST_SPECS:
        tier_manifest = EpisodeManifest(
            dataset_version=SAMPLE_DATASET_VERSION,
            task_id=tier_spec["task_id"],
            split=SplitSpec(name=tier_spec["split"]),
            warmup_start=tier_spec["warmup_start"],
            episode_start=tier_spec["episode_start"],
            episode_end=tier_spec["episode_end"],
            initial_cash=Decimal("100000"),
            universe_asset_ids=list(tier_spec["universe"]),
            cost_model_version="v0",
            scorer_version="v0",
            sandbox_image_digest="sha256:sample",
            dependency_lock_digest="sha256:lock",
            regime_labels=tier_spec["regime_labels"],
        )
        (manifests_dir / f"{tier_spec['task_id']}.json").write_text(
            tier_manifest.model_dump_json(indent=2),
            encoding="utf-8",
        )

    return version_dir


_TIER_MANIFEST_SPECS: tuple[dict[str, object], ...] = (
    {
        "task_id": "tier_t1",
        "split": "debug",
        "warmup_start": date(2019, 6, 3),
        "episode_start": T1_EPISODE_START,
        "episode_end": T1_EPISODE_END,
        "universe": T1_ASSET_IDS,
        "regime_labels": {"tier": "t1", "horizon": "60_bars", "role": "debug"},
    },
    {
        "task_id": "tier_train",
        "split": "train",
        "warmup_start": date(2023, 6, 1),
        "episode_start": TRAIN_EPISODE_START,
        "episode_end": TRAIN_EPISODE_END,
        "universe": PAIR_ASSET_IDS,
        "regime_labels": {
            "tier": "train",
            "horizon": "252_bars",
            "role": "study_packet",
        },
    },
    {
        "task_id": "tier_test",
        "split": "test",
        "warmup_start": date(2024, 6, 1),
        "episode_start": TEST_EPISODE_START,
        "episode_end": TEST_EPISODE_END,
        "universe": PAIR_ASSET_IDS,
        "regime_labels": {
            "tier": "test",
            "horizon": "120_bars",
            "role": "held_out_rollout",
        },
    },
)


def _synthetic_bar_series(
    asset_id: str,
    sessions: list[date],
) -> list[dict[str, object]]:
    """Deterministic synthetic OHLC bars for a tier asset across ``sessions``.

    Uses ``random.Random(seed=hash(asset_id))`` to produce a geometric
    Brownian motion path with a per-asset drift sampled from ``±0.0004``
    (≈ ±10%/yr) and ~1.5% daily volatility. Returns one ``DailyBarRow``-
    shaped dict per session. Two runs over the same asset_id and session
    list produce byte-identical output, which the verifier will exercise.
    """

    seed = abs(hash(asset_id)) & 0xFFFFFFFF
    rng = random.Random(seed)
    drift = (rng.random() - 0.5) * 8e-4
    base_price = 50.0 + (seed % 50)
    log_price = math.log(base_price)

    rows: list[dict[str, object]] = []
    for session in sessions:
        log_ret = drift + rng.gauss(0.0, 0.015)
        log_price += log_ret
        close_f = math.exp(log_price)
        open_jitter = 1.0 + (rng.random() - 0.5) * 0.004
        open_f = close_f * open_jitter
        hi = max(open_f, close_f) * 1.005
        lo = min(open_f, close_f) * 0.995
        vol = 500_000 + rng.randint(0, 1_000_000)
        close_d = Decimal(str(round(close_f, 4)))
        open_d = Decimal(str(round(open_f, 4)))
        high_d = Decimal(str(round(hi, 4)))
        low_d = Decimal(str(round(lo, 4)))
        rows.append(
            {
                "asset_id": asset_id,
                "session_date": session,
                "open": open_d,
                "high": high_d,
                "low": low_d,
                "close": close_d,
                "volume": vol,
                "dollar_volume": close_d * Decimal(vol),
                "available_at": _eod(session),
            },
        )
    return rows


def _bar_rows(
    asset_id: str,
    session: date,
    *,
    close: Decimal,
) -> list[dict[str, object]]:
    open_ = close - Decimal("0.25")
    high = close + Decimal("0.50")
    low = close - Decimal("0.50")
    vol = 1_000_000
    dollar = close * Decimal(vol)
    return [
        {
            "asset_id": asset_id,
            "session_date": session,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": vol,
            "dollar_volume": dollar,
            "available_at": _eod(session),
        },
    ]


def _master_row(
    asset_id: str,
    symbol: str,
    snapshot: date,
    *,
    delisting_date: date | None,
    replaces: str | None,
    replaced_by: str | None,
) -> dict[str, object]:
    return {
        "asset_id": asset_id,
        "symbol": symbol,
        "primary_exchange": "XNYS",
        "listing_date": date(2018, 1, 3),
        "delisting_date": delisting_date,
        "replaces_asset_id": replaces,
        "replaced_by_asset_id": replaced_by,
        "snapshot_date": snapshot,
        "available_at": _eod(snapshot),
    }
