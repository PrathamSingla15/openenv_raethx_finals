"""Shared helpers for the three baseline drivers."""

from __future__ import annotations

from typing import Any, Protocol


class EnvLike(Protocol):
    """Async env surface accepted by every baseline driver.

    Both ``client.TradeBenchEnv`` (HTTP) and a small in-process wrapper
    around ``server.tradebench_environment.TradeBenchEnvironment`` satisfy
    this protocol — the baselines do not care which.
    """

    async def reset(self, **kwargs: Any) -> Any: ...

    async def step(self, action: Any) -> Any: ...


def empty_baseline_result(name: str, tier: str) -> dict[str, Any]:
    """Sentinel summary used when an episode never advances (errors / abort)."""

    return {
        "name": name,
        "tier": tier,
        "rewards": [],
        "reward_breakdowns": [],
        "final_value": 0.0,
        "steps": 0,
        "ended_with_error": True,
    }


__all__ = ["EnvLike", "empty_baseline_result"]
