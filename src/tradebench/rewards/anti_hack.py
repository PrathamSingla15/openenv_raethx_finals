"""Regex-based anti-hack scanners.

Two independent surfaces:

- ``scan_forbidden_globals(text)`` inspects sandbox stdout / stderr / source
  for dangerous Python constructs that the sandbox alone cannot stop
  (reflection on ledger state, dynamic import escape, subprocess / socket).

- ``check_rules_clause(agent_output)`` inspects the agent's natural-language
  reasoning for rules-clause violations (specific historical recalls,
  phrases like "I recall", "in reality", explicit ticker-outcome pairs).

Both return ``list[str]`` of matched patterns — empty means clean. Returning
the list (rather than a bool) lets the training loop log which specific
pattern fired, which matters for debugging false positives.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

FORBIDDEN_PATTERNS: list[str] = [
    # Dynamic code execution
    r"\b__import__\s*\(",
    r"\beval\s*\(",
    r"\bexec\s*\(",
    r"\bcompile\s*\(",
    # Dunder access via getattr — usual reflection escape hatch
    r"\bgetattr\s*\([^,]+,\s*['\"]_",
    # Indirect-import / native-call helpers
    r"\bimportlib\.",
    r"\bctypes\b",
    r"\bsubprocess\.",
    r"\bsocket\.",
    # Ledger / cash tampering — private attrs on our own types
    r"(?<!\w)_cash(?!\w)",
    r"(?<!\w)_positions(?!\w)",
    r"\bPortfolioState\b.*\.__setattr__",
]

# Case-insensitive so "EVAL(" etc. also trip.
_FORBIDDEN_REGEXES: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE | re.DOTALL) for p in FORBIDDEN_PATTERNS
]

# Phrases that indicate the agent is leaning on memorized historical outcomes.
_TICKER_ALTERNATION = (
    r"(?:AAPL|MSFT|GOOGL|GOOG|AMZN|META|NVDA|TSLA|JPM|BAC|XOM|CVX|"
    r"JNJ|PFE|V|MA|DIS|NFLX|KO|PEP|WMT|HD|UNH|PG|BRK\.?B)"
)
_OUTCOME_VERB = (
    r"(?:rallied|crashed|fell|rose|surged|plunged|soared|tanked|"
    r"dropped|jumped|spiked|collapsed|recovered)"
)

RULES_CLAUSE_PATTERNS: list[str] = [
    # First-person recall phrasing
    r"\bI\s+(?:know|knew|recall|remember|remembered)\s+(?:that|what|how)\b",
    r"\bas\s+I\s+(?:recall|remember)\b",
    # Claims about what actually happened / reality
    r"\bin\s+reality\b",
    r"\bactually\s+happened\b",
    r"\bwhat\s+really\s+happened\b",
    # Ticker-outcome pair recalls (e.g. "AAPL rallied in Q2 2023")
    rf"\b{_TICKER_ALTERNATION}\s+(?:\w+\s+){{0,5}}{_OUTCOME_VERB}\b",
    # Quarter-year outcome assertions ("Q2 2023 was/ended ...")
    r"\b(?:Q[1-4])\s+(?:20\d\d)\b[^.]*?\b(?:was|were|ended|closed|finished)\b",
    # Known regime / shock names
    r"\bthe\s+(?:(?:great|global)\s+)?financial\s+crisis\b",
    r"\bCOVID\s+(?:pandemic|crash|shock|selloff)\b",
    r"\bdot[\s-]?com\s+(?:bubble|crash|bust)\b",
    r"\btariff\s+(?:war|shock)\b",
    # Direct year-outcome assertions
    r"\bin\s+20\d\d\s+(?:the\s+)?market\s+(?:crashed|rallied|surged|plunged)\b",
]

_RULES_CLAUSE_REGEXES: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE | re.DOTALL) for p in RULES_CLAUSE_PATTERNS
]


def _scan(text: str, regexes: Iterable[re.Pattern[str]]) -> list[str]:
    """Return ``pattern.pattern`` strings that produced at least one match.

    Returning ``.pattern`` rather than the raw matched text keeps potentially
    sensitive substrings out of logs while still identifying which rule fired.
    """

    if not text:
        return []
    hits: list[str] = []
    for regex in regexes:
        if regex.search(text):
            hits.append(regex.pattern)
    return hits


def scan_forbidden_globals(text: str) -> list[str]:
    """Return the forbidden-global patterns matched in ``text``.

    ``text`` is typically the concatenation of sandbox stdout, stderr and any
    source files the agent wrote into the workspace. Empty list means clean.
    """

    return _scan(text, _FORBIDDEN_REGEXES)


def check_rules_clause(agent_output: str) -> list[str]:
    """Return the rules-clause patterns matched in ``agent_output``.

    ``agent_output`` should be the agent's free-text reasoning for the bar
    (not tool arguments, not portfolio dumps). Empty list means clean.
    """

    return _scan(agent_output, _RULES_CLAUSE_REGEXES)


__all__ = [
    "FORBIDDEN_PATTERNS",
    "RULES_CLAUSE_PATTERNS",
    "check_rules_clause",
    "scan_forbidden_globals",
]


if __name__ == "__main__":
    # Seeded assertions — exercise every scanner branch that should / should
    # not fire. Any change to the pattern list must keep these green.

    # --- scan_forbidden_globals ---
    # Not flagged: we rely on the sandbox to suppress os, and plain imports
    # should not cascade to a scanner hit (avoids false positives on all
    # legitimate agent code).
    assert scan_forbidden_globals("import os\nos.system('ls')") == [], (
        "os.system should not be flagged by the forbidden-global scanner"
    )

    # Obvious hits
    assert scan_forbidden_globals("eval('1+1')") != []
    assert scan_forbidden_globals("x = __import__('os')") != []
    assert scan_forbidden_globals("exec(code)") != []
    assert scan_forbidden_globals("compile(src, '<s>', 'exec')") != []
    assert scan_forbidden_globals("getattr(obj, '__class__')") != []
    assert scan_forbidden_globals("importlib.import_module('os')") != []
    assert scan_forbidden_globals("import ctypes") != []
    assert scan_forbidden_globals("subprocess.run(['ls'])") != []
    assert scan_forbidden_globals("socket.socket()") != []
    assert scan_forbidden_globals("state._cash = Decimal('1e9')") != []
    assert scan_forbidden_globals("portfolio._positions['FAKE'] = 99999") != []
    assert (
        scan_forbidden_globals(
            "PortfolioState(cash=0).__setattr__('cash', 1e9)",
        )
        != []
    )

    # Case insensitivity
    assert scan_forbidden_globals("EVAL('hi')") != []

    # Clean benign code
    assert scan_forbidden_globals("import pandas as pd\ndf.mean()") == []
    assert (
        scan_forbidden_globals("print('positions:', positions_by_asset)") == []
    )  # non-underscore field

    # --- check_rules_clause ---
    assert check_rules_clause("I recall that AAPL rallied in Q2 2023") != []
    assert check_rules_clause("I remember what happened in 2020") != []
    assert check_rules_clause("As I recall, the market moved sharply") != []
    assert check_rules_clause("In reality the trend was upward") != []
    assert check_rules_clause("What actually happened was a rally") != []
    assert check_rules_clause("NVDA surged after earnings") != []
    assert check_rules_clause("Q2 2022 was terrible for tech") != []
    assert check_rules_clause("Remember the financial crisis of 2008") != []
    assert check_rules_clause("The COVID crash wiped out gains") != []
    assert check_rules_clause("After the dot-com bust in 2000") != []

    # Clean technical reasoning
    assert (
        check_rules_clause("Based on the 20-bar rolling mean, I propose...") == []
    )
    assert (
        check_rules_clause(
            "Compute the sample Sharpe ratio and allocate via Kelly.",
        )
        == []
    )
    assert (
        check_rules_clause("The 60-day realized vol is 18%, suggesting...") == []
    )

    print("anti_hack scanners OK")
