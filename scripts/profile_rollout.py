"""Profile one full TradeBench rollout — wall-time breakdown per bar.

Most of a rollout's wall time is LLM inference + sandbox_exec rather than
env step. This script measures the split so per-episode cost can be
extrapolated from a short trace.

Modes
-----
- ``--driver fake_advance`` (default): hammers ``advance_day`` to time the
  env-only floor.
- ``--driver fake_active``: cycles a realistic active-agent call mix
  every 3rd bar (view_portfolio + record_decision + place_order +
  advance_day) so env time is measured under load.
- ``--driver llm``: drives the env with an OpenAI-compatible LLM.
  ``--provider hf`` routes through HuggingFace Inference Providers
  (``https://router.huggingface.co/v1``, ``HF_TOKEN``); ``--provider
  openai`` uses ``OPENAI_BASE_URL`` + ``OPENAI_API_KEY``.

Usage
-----
::

    # Env-only floor on T2 (the train tier):
    uv run python scripts/profile_rollout.py --tier t2 --bars 30

    # Realistic call mix (no LLM):
    uv run python scripts/profile_rollout.py --tier t2 --bars 30 \\
        --driver fake_active

    # Real LLM rollout against HF Inference Providers (Qwen3-32B on Groq):
    export HF_TOKEN=hf_...
    uv run python scripts/profile_rollout.py --tier t1 --bars 60 \\
        --driver llm --provider hf --model "Qwen/Qwen3-32B:groq"

    # Real LLM rollout against a self-hosted vLLM endpoint:
    OPENAI_BASE_URL=http://localhost:8000/v1 OPENAI_API_KEY=local \\
        uv run python scripts/profile_rollout.py --tier t1 --bars 60 \\
        --driver llm --provider openai --model qwen3-8b-instruct

Output
------
Prints a per-category breakdown (wall time, % of total, calls, ms/call)
and writes the same payload to ``notebooks/runs/profile_<tier>.json``
for later plotting.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from server.tradebench_environment import TradeBenchEnvironment

try:
    from models import TradeAction
except ImportError:  # pragma: no cover
    from tradebench.models import TradeAction  # type: ignore[no-redef]


_DEFAULT_RUNS_DIR = Path(__file__).resolve().parent.parent / "notebooks" / "runs"


@dataclass
class _Timer:
    """Per-bucket cumulative timer (calls, total seconds)."""

    name: str
    total_seconds: float = 0.0
    calls: int = 0
    samples_seconds: list[float] = field(default_factory=list)

    def add(self, seconds: float) -> None:
        self.total_seconds += seconds
        self.calls += 1
        self.samples_seconds.append(seconds)

    def summary(self) -> dict[str, float | int]:
        if not self.calls:
            return {"calls": 0, "total_s": 0.0, "mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}
        sorted_s = sorted(self.samples_seconds)
        return {
            "calls": self.calls,
            "total_s": round(self.total_seconds, 4),
            "mean_ms": round(1000 * self.total_seconds / self.calls, 2),
            "p50_ms": round(1000 * sorted_s[len(sorted_s) // 2], 2),
            "p95_ms": round(1000 * sorted_s[min(len(sorted_s) - 1, int(0.95 * len(sorted_s)))], 2),
        }


def _action_for_fake_advance(_step: int) -> TradeAction:
    return TradeAction(action_type="advance_day")


def _action_for_fake_active(step: int) -> TradeAction:
    """Cycle a realistic active-agent pattern every 3 bars."""
    cycle = step % 3
    if cycle == 0:
        return TradeAction(action_type="view_portfolio")
    if cycle == 1:
        return TradeAction(
            action_type="record_decision",
            payload={
                "regime_label": "neutral",
                "edge_summary": "synthetic profile call",
                "intended_exposure": "0.50",
                "top_convictions": [],
                "uncertainty": "medium",
            },
        )
    return TradeAction(action_type="advance_day")


async def _llm_action(
    client: Any,
    model: str,
    obs: Any,
    prompt: str,
    timer: _Timer,
) -> TradeAction:
    """One chat-completion call timed under ``timer``."""
    request_text = (
        f"Current observation:\n{_compact_obs(obs)}\n\n"
        "Return ONLY a single JSON object with keys 'action_type' and 'payload'."
    )
    t0 = time.perf_counter()
    try:
        resp = await asyncio.to_thread(
            client.chat.completions.create,
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": request_text},
            ],
            # Headroom for Qwen3 thinking-mode <think> blocks plus the answer.
            max_tokens=2048,
        )
        raw = resp.choices[0].message.content or ""
    except Exception as exc:  # noqa: BLE001 — profile, not eval
        print(f"  llm call failed: {exc}", file=sys.stderr)
        raw = ""
    timer.add(time.perf_counter() - t0)
    return _parse_action(raw)


_THINK_RE = __import__("re").compile(r"<think>.*?</think>", __import__("re").DOTALL)
_FENCED_JSON_RE = __import__("re").compile(
    r"```(?:json)?\s*(\{.*?\})\s*```", __import__("re").DOTALL
)
_LOOSE_OBJECT_RE = __import__("re").compile(r"\{[^{}]*?\"action_type\".*?\}", __import__("re").DOTALL)


def _parse_action(raw: str) -> TradeAction:
    """Best-effort JSON parse; falls back to ``advance_day`` on garbage.

    Strips Qwen3-style ``<think>...</think>`` blocks before parsing so the
    answer-JSON is what we read.
    """
    raw = _THINK_RE.sub("", raw).strip()
    candidates: list[str] = []
    candidates.extend(_FENCED_JSON_RE.findall(raw))
    candidates.extend(_LOOSE_OBJECT_RE.findall(raw))
    if not candidates and raw.startswith("{"):
        candidates.append(raw)
    for candidate in reversed(candidates):
        try:
            d = json.loads(candidate)
        except (ValueError, json.JSONDecodeError):
            continue
        if isinstance(d, dict) and "action_type" in d:
            return TradeAction(
                action_type=d.get("action_type", "advance_day"),
                payload=d.get("payload") or {},
            )
    return TradeAction(action_type="advance_day")


def _compact_obs(obs: Any) -> str:

    parts = [
        f"date={obs.current_date}  bars_remaining={obs.bars_remaining}  phase={obs.phase}",
        f"value={obs.portfolio_value:.2f}  cash={obs.cash:.2f}",
    ]
    if obs.positions:
        parts.append(f"positions={dict(obs.positions)}")
    if obs.tool_output:
        parts.append("---\n" + obs.tool_output[:600])
    return "\n".join(parts)


_BASE_PROMPT = (
    "You are operating inside a TradeBench trading episode. Maximize "
    "long-run log-wealth. Base every decision on systematic models using "
    "ONLY data available at the current simulation date. Do not recall "
    "memorized historical events. Respond with JSON only, e.g. "
    '{"action_type":"advance_day","payload":{}}.'
)


async def profile_rollout(
    *,
    tier: str,
    bars: int,
    driver: str,
    model: str | None,
    provider: str = "openai",
) -> dict[str, Any]:
    env = TradeBenchEnvironment()
    obs = env.reset(task_tier=tier)
    print(f"reset: tier={tier} bars_planned={bars} initial_value={obs.portfolio_value:.2f}")

    env_timer = _Timer("env.step")
    sandbox_timer = _Timer("env.sandbox_exec")
    llm_timer = _Timer("llm.completion")
    rewards: list[float] = []

    llm_client = None
    if driver == "llm":
        try:
            from openai import OpenAI  # type: ignore[import-not-found]
        except ImportError:
            print("ERROR: --driver llm requires `pip install openai`.", file=sys.stderr)
            sys.exit(2)
        if provider == "hf":
            base_url = os.environ.get(
                "HF_ROUTER_URL",
                "https://router.huggingface.co/v1",
            )
            api_key = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                print(
                    "ERROR: --provider hf requires HF_TOKEN env var "
                    "(create at https://huggingface.co/settings/tokens).",
                    file=sys.stderr,
                )
                sys.exit(2)
        else:
            base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
            api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
        print(f"llm: provider={provider}  base_url={base_url}  model={model}")
        llm_client = OpenAI(api_key=api_key, base_url=base_url)

    t_start = time.perf_counter()
    for step in range(bars):
        if obs.done:
            break
        if driver == "fake_advance":
            action = _action_for_fake_advance(step)
        elif driver == "fake_active":
            action = _action_for_fake_active(step)
        elif driver == "llm":
            action = await _llm_action(
                llm_client,
                model or "gpt-4o-mini",
                obs,
                _BASE_PROMPT,
                llm_timer,
            )
        else:
            raise ValueError(f"unknown driver {driver!r}")

        bucket = sandbox_timer if action.action_type == "sandbox_exec" else env_timer
        t0 = time.perf_counter()
        obs = env.step(action)
        bucket.add(time.perf_counter() - t0)
        rewards.append(float(obs.reward or 0.0))

    wall = time.perf_counter() - t_start
    env.close()

    summary = {
        "tier": tier,
        "driver": driver,
        "model": model if driver == "llm" else None,
        "bars_completed": len(rewards),
        "wall_seconds": round(wall, 3),
        "ms_per_bar": round(1000 * wall / max(1, len(rewards)), 1),
        "total_reward": round(sum(rewards), 4),
        "buckets": {
            "llm.completion": llm_timer.summary(),
            "env.sandbox_exec": sandbox_timer.summary(),
            "env.step (other)": env_timer.summary(),
        },
        "bucket_share_pct": _bucket_share(env_timer, sandbox_timer, llm_timer, total=wall),
    }
    return summary


def _bucket_share(*timers: _Timer, total: float) -> dict[str, float]:
    if total <= 0:
        return {t.name: 0.0 for t in timers}
    return {
        t.name: round(100 * t.total_seconds / total, 1)
        for t in timers
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tier", default="test", choices=("t1", "train", "test"))
    ap.add_argument("--bars", type=int, default=30)
    ap.add_argument(
        "--driver",
        default="fake_advance",
        choices=("fake_advance", "fake_active", "llm"),
    )
    ap.add_argument(
        "--provider",
        default="openai",
        choices=("openai", "hf"),
        help=(
            "LLM provider for --driver llm. 'hf' routes through HuggingFace "
            "Inference Providers (https://router.huggingface.co/v1, HF_TOKEN); "
            "'openai' uses OPENAI_BASE_URL + OPENAI_API_KEY. Default: openai."
        ),
    )
    ap.add_argument(
        "--model",
        default=None,
        help=(
            "Model name for --driver llm. For --provider hf, append a "
            "policy/provider suffix, e.g. 'Qwen/Qwen3-32B:groq' or "
            "'Qwen/Qwen3-32B:fastest' / ':cheapest'. Default: gpt-4o-mini "
            "(openai) / Qwen/Qwen3-32B:groq (hf)."
        ),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="JSON output path (default: notebooks/runs/profile_<tier>_<driver>.json)",
    )
    args = ap.parse_args()
    if args.driver == "llm" and args.model is None:
        args.model = (
            "Qwen/Qwen3-32B:groq" if args.provider == "hf" else "gpt-4o-mini"
        )

    summary = asyncio.run(
        profile_rollout(
            tier=args.tier,
            bars=args.bars,
            driver=args.driver,
            model=args.model,
            provider=args.provider,
        ),
    )

    print()
    print(f"=== PROFILE SUMMARY ({summary['tier']}, {summary['driver']}) ===")
    print(f"  bars completed: {summary['bars_completed']}")
    print(f"  wall: {summary['wall_seconds']:.2f}s  ({summary['ms_per_bar']} ms/bar)")
    print(f"  total reward: {summary['total_reward']:+.4f}")
    print(f"  share of wall:")
    for name, pct in summary["bucket_share_pct"].items():
        print(f"    {name:<22s}  {pct:5.1f}%")
    print(f"  per-bucket detail:")
    for name, det in summary["buckets"].items():
        if det["calls"] == 0:
            continue
        print(
            f"    {name:<22s}  calls={det['calls']:>4d}  "
            f"mean={det['mean_ms']:>7.1f} ms  "
            f"p50={det['p50_ms']:>7.1f} ms  p95={det['p95_ms']:>7.1f} ms",
        )

    out_path = args.out or (
        _DEFAULT_RUNS_DIR / f"profile_{args.tier}_{args.driver}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
