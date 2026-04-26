"""Pytest coverage for the date-gate decorator + helpers."""

from __future__ import annotations

from datetime import date

import pytest

from tradebench.environment.date_gate import (
    LookaheadViolation,
    check_date_bound,
    gate_data_tool,
    silently_truncate,
)


def test_check_date_bound_accepts_past() -> None:
    check_date_bound(date(2020, 1, 1), date(2020, 3, 15))


def test_check_date_bound_accepts_equal() -> None:
    check_date_bound(date(2020, 3, 15), date(2020, 3, 15))


def test_check_date_bound_rejects_future() -> None:
    with pytest.raises(LookaheadViolation):
        check_date_bound(date(2020, 4, 1), date(2020, 3, 15))


def test_check_date_bound_string_inputs() -> None:
    check_date_bound("2020-03-10", "2020-03-15")
    with pytest.raises(LookaheadViolation):
        check_date_bound("2020-04-01", "2020-03-15")


def test_silently_truncate_clamps_future() -> None:
    assert silently_truncate("2020-04-01", date(2020, 3, 15)) == date(2020, 3, 15)


def test_silently_truncate_passes_past() -> None:
    assert silently_truncate("2020-02-01", date(2020, 3, 15)) == date(2020, 2, 1)


def test_gate_data_tool_decorator_passes_valid() -> None:
    today = date(2020, 3, 15)

    @gate_data_tool(lambda: today)
    def fake(asset_ids, as_of_date, lookback_days=5):
        return (asset_ids, str(as_of_date), lookback_days)

    assert fake(["AAPL"], as_of_date="2020-03-10") == (["AAPL"], "2020-03-10", 5)


def test_gate_data_tool_decorator_rejects_future() -> None:
    today = date(2020, 3, 15)

    @gate_data_tool(lambda: today)
    def fake(as_of_date):
        return as_of_date

    with pytest.raises(LookaheadViolation):
        fake(as_of_date="2020-04-01")


def test_gate_data_tool_requires_param_to_exist() -> None:
    today = date(2020, 3, 15)
    with pytest.raises(TypeError):

        @gate_data_tool(lambda: today, as_of_param="bogus_arg")
        def fake(as_of_date):  # missing 'bogus_arg'
            return as_of_date


def test_gate_data_tool_missing_argument_fails_closed() -> None:
    today = date(2020, 3, 15)

    @gate_data_tool(lambda: today)
    def fake(as_of_date=None):
        return as_of_date

    with pytest.raises(LookaheadViolation):
        fake()  # no as_of_date supplied
