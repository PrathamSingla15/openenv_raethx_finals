"""Pytest coverage for the anti-hack scanners.

Mirrors the assertions in ``rewards/anti_hack.py``'s ``__main__`` block,
plus a few additional regressions for false-positive cases the regex
patterns must not flag.
"""

from __future__ import annotations

import pytest

from tradebench.rewards.anti_hack import check_rules_clause, scan_forbidden_globals


@pytest.mark.parametrize(
    "snippet",
    [
        "eval('1+1')",
        "x = __import__('os')",
        "exec(code)",
        "compile(src, '<s>', 'exec')",
        "getattr(obj, '__class__')",
        "importlib.import_module('os')",
        "import ctypes",
        "subprocess.run(['ls'])",
        "socket.socket()",
        "state._cash = Decimal('1e9')",
        "portfolio._positions['FAKE'] = 99999",
        "PortfolioState(cash=0).__setattr__('cash', 1e9)",
        "EVAL('hi')",  # case insensitivity
    ],
)
def test_forbidden_global_hits(snippet: str) -> None:
    assert scan_forbidden_globals(snippet), snippet


@pytest.mark.parametrize(
    "snippet",
    [
        "import pandas as pd\ndf.mean()",
        "print('positions:', positions_by_asset)",
        "import os\nos.system('ls')",  # os.system relies on sandbox, not flagged
    ],
)
def test_forbidden_global_clean(snippet: str) -> None:
    assert scan_forbidden_globals(snippet) == []


@pytest.mark.parametrize(
    "snippet",
    [
        "I recall that AAPL rallied in Q2 2023",
        "I remember what happened in 2020",
        "As I recall, the market moved sharply",
        "In reality the trend was upward",
        "What actually happened was a rally",
        "NVDA surged after earnings",
        "Q2 2022 was terrible for tech",
        "Remember the financial crisis of 2008",
        "The COVID crash wiped out gains",
        "After the dot-com bust in 2000",
    ],
)
def test_rules_clause_hits(snippet: str) -> None:
    assert check_rules_clause(snippet), snippet


@pytest.mark.parametrize(
    "snippet",
    [
        "Based on the 20-bar rolling mean, I propose a long position.",
        "Compute the sample Sharpe ratio and allocate via Kelly.",
        "The 60-day realized vol is 18%, suggesting low exposure.",
    ],
)
def test_rules_clause_clean(snippet: str) -> None:
    assert check_rules_clause(snippet) == []


def test_empty_input_returns_empty() -> None:
    assert scan_forbidden_globals("") == []
    assert check_rules_clause("") == []
