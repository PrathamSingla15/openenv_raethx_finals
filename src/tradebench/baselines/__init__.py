"""Reference strategies for the TradeBench env.

Three deterministic baselines used to draw the *baseline-vs-trained* line on
the hackathon's improvement-evidence plot. Each baseline is an async coroutine
that drives the env end-to-end and returns a structured summary (per-bar
rewards, per-bar 7-component breakdowns, terminal portfolio value):

- :func:`run_cash_baseline` — never buys; pure cash-only floor
- :func:`run_equal_weight_baseline` — 1/N exposure, one-shot allocation, hold
- :func:`run_kelly_baseline` — Kelly-fraction (0.5) × 1/N, one-shot, hold

The baselines deliberately don't call ``sandbox_exec`` — they exercise only
the order-management surface, so the resulting reward curves represent the
*data-blind* lower bound. A trained agent that uses the sandbox should
beat these on T1.

Driver functions accept any object exposing async ``reset(*, task_tier=...)``
and ``step(action) -> StepResult`` methods — both the openenv-core HTTP
client (``client.TradeBenchEnv``) and an in-process wrapper around the
``server.tradebench_environment.TradeBenchEnvironment`` work without
adapters. See ``inference.py:--baseline`` for the production driver.
"""

from .cash import run_cash_baseline
from .equal_weight import run_equal_weight_baseline
from .kelly import run_kelly_baseline

BASELINES = {
    "cash": run_cash_baseline,
    "equal_weight": run_equal_weight_baseline,
    "kelly": run_kelly_baseline,
}

__all__ = [
    "BASELINES",
    "run_cash_baseline",
    "run_equal_weight_baseline",
    "run_kelly_baseline",
]
