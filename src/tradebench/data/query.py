"""Read-only DuckDB query helpers with point-in-time filters."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import TracebackType
from typing import Any, cast

import duckdb
import pandas as pd  # type: ignore[import-untyped]

from tradebench.data.catalog import DatasetCatalog
from tradebench.data.models import (
    CorporateActionKind,
    CorporateActionRow,
    ensure_utc,
    utc_end_of_day,
)
from tradebench.episodes.models import EpisodeManifest


def _date_to_pit_bound(d: date) -> datetime:
    """Inclusive upper bound for PIT queries keyed by a session calendar date."""

    return utc_end_of_day(d)


def _sql_string_literal(path: Path) -> str:
    escaped = str(path.resolve()).replace("'", "''")
    return f"'{escaped}'"


def _sql_quoted_glob(glob_pattern: str) -> str:
    escaped = glob_pattern.replace("'", "''")
    return f"'{escaped}'"


def _coerce_date_columns(df: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for column in columns:
        if column not in out.columns:
            continue
        parsed = pd.to_datetime(out[column], errors="coerce")

        def _to_py_date(value: object) -> date | None:
            if value is None or (isinstance(value, float) and pd.isna(value)):
                return None
            ts = pd.Timestamp(value)
            if pd.isna(ts):
                return None
            return cast(date, ts.date())

        out[column] = parsed.map(_to_py_date)
    return out


class PitQueryService:
    """
    Small read-only DuckDB facade over versioned Parquet tables.

    Every exposure path enforces ``available_at <= pit_bound`` for the supplied
    ``as_of`` / session date (end-of-day UTC for date-keyed queries).
    """

    def __init__(self, catalog: DatasetCatalog) -> None:
        catalog.validate_layout()
        self._catalog = catalog
        self._paths = catalog.paths()
        self._con = duckdb.connect(database=":memory:")
        self._closed = False
        self._register_views()

    def __enter__(self) -> PitQueryService:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._con.close()
        self._closed = True

    def _register_views(self) -> None:
        p = self._paths
        self._con.execute(
            "CREATE VIEW asset_master AS "
            f"SELECT * FROM read_parquet({_sql_string_literal(p.asset_master)});",
        )
        self._con.execute(
            "CREATE VIEW daily_bars AS "
            f"SELECT * FROM read_parquet({_sql_quoted_glob(p.daily_bars_glob)});",
        )
        self._con.execute(
            "CREATE VIEW corporate_actions AS "
            f"SELECT * FROM read_parquet({_sql_string_literal(p.corporate_actions)});",
        )
        self._con.execute(
            "CREATE VIEW fundamentals_pti AS "
            f"SELECT * FROM read_parquet({_sql_string_literal(p.fundamentals_pti)});",
        )
        self._con.execute(
            "CREATE VIEW calendar AS "
            f"SELECT * FROM read_parquet({_sql_string_literal(p.calendar)});",
        )

    @staticmethod
    def _bar_window_dates(end_date: date, lookback_days: int) -> tuple[date, date]:
        if lookback_days < 1:
            msg = "lookback_days must be >= 1"
            raise ValueError(msg)
        start = end_date - timedelta(days=lookback_days - 1)
        return start, end_date

    def load_universe(self, as_of: date, manifest: EpisodeManifest) -> list[str]:
        """
        Return manifest universe members with a PIT-visible master row on ``as_of``.

        A row qualifies when ``snapshot_date <= as_of`` and
        ``available_at <= utc_end_of_day(as_of)``.
        """

        if not manifest.universe_asset_ids:
            return []

        pit = _date_to_pit_bound(as_of)
        placeholders = ", ".join(["?"] * len(manifest.universe_asset_ids))
        sql = f"""
            SELECT DISTINCT asset_id
            FROM asset_master
            WHERE asset_id IN ({placeholders})
              AND snapshot_date <= ?::DATE
              AND available_at <= ?::TIMESTAMPTZ
            ORDER BY asset_id
        """
        params: list[Any] = [*manifest.universe_asset_ids, as_of, pit]
        rows = self._con.execute(sql, params).fetchall()
        return [str(r[0]) for r in rows]

    def get_bars(
        self,
        asset_ids: list[str],
        end_date: date,
        lookback_days: int,
    ) -> pd.DataFrame:
        """
        Return OHLCV bars whose ``session_date`` falls in the inclusive calendar window
        ``[end_date - lookback_days + 1, end_date]`` and that are PIT-visible at
        ``utc_end_of_day(end_date)``.
        """

        if not asset_ids:
            return pd.DataFrame()

        start, end = self._bar_window_dates(end_date, lookback_days)
        pit = _date_to_pit_bound(end_date)
        placeholders = ", ".join(["?"] * len(asset_ids))
        sql = f"""
            SELECT asset_id, session_date, open, high, low, close, volume,
                   dollar_volume, available_at
            FROM daily_bars
            WHERE asset_id IN ({placeholders})
              AND session_date >= ?::DATE
              AND session_date <= ?::DATE
              AND available_at <= ?::TIMESTAMPTZ
            ORDER BY asset_id, session_date
        """
        params: list[Any] = [*asset_ids, start, end, pit]
        df = self._con.execute(sql, params).df()
        return _coerce_date_columns(df, ("session_date",))

    def get_corporate_actions(
        self,
        asset_ids: list[str],
        session_date: date,
    ) -> list[CorporateActionRow]:
        """
        Corporate actions whose ``effective_date`` equals ``session_date`` and that are
        PIT-visible at ``utc_end_of_day(session_date)``.
        """

        if not asset_ids:
            return []

        pit = _date_to_pit_bound(session_date)
        placeholders = ", ".join(["?"] * len(asset_ids))
        sql = f"""
            SELECT asset_id, action_type, effective_date, ex_date,
                   split_from, split_to, dividend_amount, new_symbol,
                   metadata, available_at
            FROM corporate_actions
            WHERE asset_id IN ({placeholders})
              AND effective_date = ?::DATE
              AND available_at <= ?::TIMESTAMPTZ
            ORDER BY asset_id, action_type
        """
        params: list[Any] = [*asset_ids, session_date, pit]
        df = self._con.execute(sql, params).df()
        rows: list[CorporateActionRow] = []
        for record in df.to_dict(orient="records"):
            rows.append(_corporate_action_from_record(record))
        return rows

    def get_fundamentals(
        self,
        asset_ids: list[str],
        as_of: datetime,
    ) -> pd.DataFrame:
        """Fundamental snapshots PIT-visible at ``as_of`` (UTC-normalized)."""

        if not asset_ids:
            return pd.DataFrame()

        as_of_utc = ensure_utc(as_of)
        placeholders = ", ".join(["?"] * len(asset_ids))
        sql = f"""
            SELECT asset_id, fiscal_period_start, fiscal_period_end,
                   revenue, net_income, shares_outstanding, available_at
            FROM fundamentals_pti
            WHERE asset_id IN ({placeholders})
              AND available_at <= ?::TIMESTAMPTZ
            ORDER BY asset_id, fiscal_period_end, available_at
        """
        params: list[Any] = [*asset_ids, as_of_utc]
        df = self._con.execute(sql, params).df()
        return _coerce_date_columns(df, ("fiscal_period_start", "fiscal_period_end"))

    def next_session_after(self, session_date: date) -> date | None:
        """Next session strictly after ``session_date`` (from catalog calendar)."""

        row = self._con.execute(
            "SELECT MIN(session_date) AS n FROM calendar WHERE session_date > ?::DATE",
            [session_date],
        ).fetchone()
        if row is None or row[0] is None:
            return None
        val = row[0]
        if isinstance(val, date):
            return val
        return cast(date, pd.Timestamp(val).date())

    def get_symbol_to_asset_id(
        self,
        as_of: date,
        asset_ids: list[str],
    ) -> dict[str, str]:
        """
        Map listing symbol -> ``asset_id`` (latest PIT-visible master row per asset).

        Raises ``ValueError`` if two assets share the same symbol in this slice.
        """

        if not asset_ids:
            return {}

        pit = _date_to_pit_bound(as_of)
        placeholders = ", ".join(["?"] * len(asset_ids))
        sql = f"""
            WITH ranked AS (
              SELECT asset_id, symbol,
                ROW_NUMBER() OVER (
                  PARTITION BY asset_id ORDER BY snapshot_date DESC
                ) AS rn
              FROM asset_master
              WHERE asset_id IN ({placeholders})
                AND snapshot_date <= ?::DATE
                AND available_at <= ?::TIMESTAMPTZ
            )
            SELECT asset_id, symbol FROM ranked WHERE rn = 1
        """
        params: list[Any] = [*asset_ids, as_of, pit]
        df = self._con.execute(sql, params).df()
        out: dict[str, str] = {}
        for rec in df.to_dict(orient="records"):
            sym = str(rec["symbol"])
            aid = str(rec["asset_id"])
            existing = out.get(sym)
            if existing is not None and existing != aid:
                msg = f"ambiguous symbol {sym!r} maps to multiple asset_ids"
                raise ValueError(msg)
            out[sym] = aid
        return out


def _corporate_action_from_record(record: dict[str, Any]) -> CorporateActionRow:
    meta = record.get("metadata")
    if meta is None or (isinstance(meta, float) and pd.isna(meta)):
        meta_dict: dict[str, Any] = {}
    elif isinstance(meta, str):
        meta_dict = json.loads(meta) if meta else {}
    elif hasattr(meta, "as_py"):
        meta_dict = cast(dict[str, Any], dict(meta.as_py()))
    elif isinstance(meta, dict):
        meta_dict = meta
    else:
        meta_dict = {}

    def _dec(v: Any) -> Decimal | None:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        return Decimal(str(v))

    eff = record["effective_date"]
    if isinstance(eff, datetime):
        eff_d = eff.date()
    elif isinstance(eff, date):
        eff_d = eff
    else:
        eff_d = cast(date, pd.Timestamp(eff).date())

    ex = record.get("ex_date")
    ex_d: date | None = None
    if ex is not None and not (isinstance(ex, float) and pd.isna(ex)):
        if isinstance(ex, datetime):
            ex_d = ex.date()
        elif isinstance(ex, date):
            ex_d = ex
        else:
            ex_d = cast(date, pd.Timestamp(ex).date())

    return CorporateActionRow(
        asset_id=str(record["asset_id"]),
        action_type=CorporateActionKind(str(record["action_type"])),
        effective_date=eff_d,
        ex_date=ex_d,
        split_from=_dec(record.get("split_from")),
        split_to=_dec(record.get("split_to")),
        dividend_amount=_dec(record.get("dividend_amount")),
        new_symbol=(
            None
            if record.get("new_symbol") is None or pd.isna(record.get("new_symbol"))
            else str(record["new_symbol"])
        ),
        metadata=meta_dict,
        available_at=ensure_utc(record["available_at"]),
    )
