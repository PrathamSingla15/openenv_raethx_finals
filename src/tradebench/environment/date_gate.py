"""Tool-level date gating for TradeBench look-ahead prevention.

The SQL layer enforces point-in-time availability via ``available_at <= ?``
in :mod:`tradebench.data.query`. This module catches future-dated tool
requests at dispatch time, before they ever reach DuckDB.

Two enforcement modes are provided:

* :func:`check_date_bound` / :func:`gate_data_tool` — *hard refusal*.
  Raises :class:`LookaheadViolation` when ``as_of > current``.
* :func:`silently_truncate` — returns ``min(requested, current)`` for
  data-read tools where a best-effort answer is preferable to an error.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from datetime import date, datetime
from typing import Any, TypeVar

__all__ = [
    "LookaheadViolation",
    "check_date_bound",
    "silently_truncate",
    "gate_data_tool",
]


F = TypeVar("F", bound=Callable[..., Any])


class LookaheadViolation(RuntimeError):
    """Raised when a tool request exceeds the current simulation date."""


def _coerce_date(value: date | str | datetime) -> date:
    """Normalize ``date``, ``datetime``, or ISO-8601 string → ``date``."""

    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            msg = f"could not parse date from string {value!r}: {exc}"
            raise LookaheadViolation(msg) from exc
    msg = f"unsupported date value: {value!r} (type {type(value).__name__})"
    raise LookaheadViolation(msg)


def check_date_bound(
    requested: date | str | datetime,
    current: date | str | datetime,
) -> None:
    """Raise :class:`LookaheadViolation` if ``requested > current``.

    String inputs are normalized to :class:`datetime.date` via ISO-8601
    parsing. Mismatched / unparseable inputs raise
    :class:`LookaheadViolation` as well — the gate fails closed.
    """

    req = _coerce_date(requested)
    cur = _coerce_date(current)
    if req > cur:
        msg = (
            f"Look-ahead violation: requested as_of_date {req.isoformat()} is "
            f"beyond the current simulation date {cur.isoformat()}. "
            "Only past/current dates are observable inside an episode."
        )
        raise LookaheadViolation(msg)


def silently_truncate(
    requested: date | str | datetime,
    current: date | str | datetime,
) -> date:
    """Return ``min(requested, current)`` (post-normalization).

    Intended for data-read tools where a best-effort truncation is
    preferable to an error.
    """

    req = _coerce_date(requested)
    cur = _coerce_date(current)
    return req if req <= cur else cur


def gate_data_tool(
    get_current_date: Callable[[], date],
    *,
    as_of_param: str = "as_of_date",
) -> Callable[[F], F]:
    """Decorator factory: reject tool calls with ``as_of_date > current``.

    The wrapped callable must accept ``as_of_date`` (or the name passed
    via ``as_of_param``) as either a positional or keyword argument. On
    every invocation, the decorator resolves ``current_date`` via the
    injected ``get_current_date`` callable (so the gate follows the
    simulation clock without a global) and raises
    :class:`LookaheadViolation` on violation.

    Parameters
    ----------
    get_current_date:
        Zero-arg callable returning the session's current date. Injected
        so tests can pass ``lambda: date(2020, 1, 15)`` without wiring a
        real session.
    as_of_param:
        Name of the keyword / positional arg carrying the requested
        date. Defaults to ``"as_of_date"``.
    """

    def decorator(func: F) -> F:
        sig = inspect.signature(func)
        if as_of_param not in sig.parameters:
            msg = (
                f"gate_data_tool: function {func.__qualname__!r} has no "
                f"parameter named {as_of_param!r}; cannot install gate."
            )
            raise TypeError(msg)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            bound = sig.bind_partial(*args, **kwargs)
            requested = bound.arguments.get(as_of_param)
            if requested is None:
                msg = (
                    f"{func.__qualname__}: required argument "
                    f"{as_of_param!r} is missing; date-gated tools must "
                    "accept an explicit as_of_date."
                )
                raise LookaheadViolation(msg)
            check_date_bound(requested, get_current_date())
            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


if __name__ == "__main__":
    # Smoke test: one good call, one bad call.
    today = date(2020, 3, 15)

    @gate_data_tool(lambda: today)
    def fake_view_bars(
        asset_ids: list[str],
        as_of_date: date | str,
        lookback_days: int = 5,
    ) -> dict[str, Any]:
        return {
            "asset_ids": asset_ids,
            "as_of_date": str(as_of_date),
            "lookback_days": lookback_days,
        }

    good = fake_view_bars(["AAPL"], as_of_date="2020-03-10", lookback_days=3)
    print("good call OK:", good)

    try:
        fake_view_bars(["AAPL"], as_of_date="2020-04-01")
    except LookaheadViolation as exc:
        print("bad call correctly rejected:", exc)
    else:
        raise SystemExit("EXPECTED LookaheadViolation WAS NOT RAISED")

    truncated = silently_truncate("2020-04-01", today)
    assert truncated == today, truncated
    passed = silently_truncate("2020-02-01", today)
    assert passed == date(2020, 2, 1), passed
    print("silently_truncate OK:", truncated, passed)
