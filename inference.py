"""Baseline inference script for the TradeBench OpenEnv environment.

Two operating modes share the same env client + log shape:

- **LLM mode (default)** drives an OpenAI-compatible model through a short
  episode on each tier (T1/T2/T3).
- **Baseline mode** (``--baseline {cash,equal_weight,kelly}``) drives the
  env through deterministic reference strategies — used to generate the
  ``baseline_vs_trained.png`` improvement-evidence plot.

Both modes emit the same ``[START]`` / ``[STEP]`` / ``[END]`` lines so the
hackathon auto-validator parses either output identically.

Environment variables:
    API_BASE_URL   LLM endpoint. If unset and HF_TOKEN is present, defaults to
                   HuggingFace Inference Providers router
                   (https://router.huggingface.co/v1). Otherwise OpenAI.
    MODEL_NAME     Model identifier. Default: ``Qwen/Qwen3-32B:groq`` when HF
                   routing is detected, ``gpt-4o-mini`` otherwise.
                   For HF routing you can append a provider/policy suffix:
                       Qwen/Qwen3-32B:groq      (specific provider)
                       Qwen/Qwen3-32B:fastest   (auto-pick fastest)
                       Qwen/Qwen3-32B:cheapest  (auto-pick cheapest)
    OPENAI_API_KEY API key. HF_TOKEN is accepted as a fallback (and is the
                   right token to use when API_BASE_URL points at HF router).
    ENV_URL        Optional: URL of a running TradeBench server.
    LOCAL_IMAGE_NAME  Optional: Docker image for ``from_docker_image``.
    MAX_STEPS_PER_TASK  Optional: override the per-task step cap (default 12).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from typing import Any, List, Optional

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - import-time guard
    OpenAI = None  # type: ignore[assignment,misc]

try:
    from tradebench_env import TradeBenchEnv, TradeAction  # type: ignore[assignment]
except ImportError:
    try:
        from client import TradeBenchEnv  # type: ignore[assignment]
        from models import TradeAction  # type: ignore[assignment]
    except ImportError:
        TradeBenchEnv = None  # type: ignore[assignment,misc]
        TradeAction = None  # type: ignore[assignment,misc]

try:
    from tradebench.baselines import BASELINES
except ImportError:  # pragma: no cover - dev fallback when src isn't on path
    BASELINES = {}  # type: ignore[assignment]

try:
    from inference_logger import RunLogger, make_run_id
except ImportError:  # pragma: no cover - dev fallback when path is unusual
    RunLogger = None  # type: ignore[assignment,misc]
    make_run_id = None  # type: ignore[assignment]

ARTIFACTS_ROOT = os.environ.get(
    "TRADEBENCH_ARTIFACTS_ROOT",
    str((__import__("pathlib").Path(__file__).resolve().parent / "artifacts")),
)
TRAIN_STUDY_RETRIES = int(os.environ.get("TRAIN_STUDY_RETRIES", "3"))


BENCHMARK = "tradebench"
TASK_IDS = ["train", "test"]

HF_SPACE_URL = "https://yobro4619-tradebench.hf.space"

_DEFAULT_MAX_STEPS = int(os.environ.get("MAX_STEPS_PER_TASK", "1200"))

SYSTEM_PROMPT = """\
You are a systematic quantitative trading agent operating inside the
TradeBench reinforcement-learning environment. Read this entire spec
once before responding. The most common failure modes are addressed by
name in the COMMON MISTAKES and WORKED EXAMPLE sections — do not skip
them.

# 1. THE GAME

- You manage a $100,000 portfolio across a sequence of trading days
  (60 / 120 / 252 days depending on the tier).
- Each day the environment surfaces an observation. You issue ONE tool
  call. The environment returns the next observation.
- Your reward each bar is the CHANGE IN LOG-WEALTH ``log(V_{t+1}/V_t)``
  PLUS bounded regularizers (Sharpe bonus, drawdown penalty, turnover
  penalty, concentration penalty, rules-clause penalty, hack penalty).
- Maximizing the cumulative reward = maximizing the log of (final
  portfolio value / starting portfolio value). Compounding matters.
- The episode ends when ``bars_remaining`` reaches 0 OR ``done`` is true.

# 2. WORLD MODEL — read this section carefully

- THE SIMULATION CLOCK ADVANCES ONLY WHEN YOU CALL ``advance_day``.
  All other tools (view_*, record_decision, place_order, cancel_order,
  sandbox_exec) leave the clock fixed.
- ORDERS FILL AT THE NEXT SESSION OPEN, NOT IMMEDIATELY. Submitting
  ``place_order`` on day T queues the order. The fill is settled the
  next time you call ``advance_day`` — at the OPEN price of day T+1
  (with deterministic slippage and fees).
- ALL DATA IS POINT-IN-TIME. You can never see the future. Querying
  data with ``as_of_date > current_date`` is rejected. The DuckDB layer
  enforces ``available_at <= current_date`` on every read.
- THE SANDBOX HAS NO INTERNET. ``sandbox_exec`` runs Python/shell in
  an isolated worker — no network access, read-only access to a copy
  of the bars/master/calendar parquet files, read-write workspace dir.
- THERE IS NO LIVE TICKER NEWS. You see prices and volumes, nothing
  else. Build statistical edges from the price series itself.

# 3. UNIVERSE — what you can actually trade

The tradable universe is a small set of OPAQUE ALIASES:
``tier_t1_a01``, ``tier_t1_a02``, ..., ``tier_t3_a20`` (depending on
tier). These are NOT real ticker symbols.

NEVER issue an order against a real ticker. There is no ``SPY``,
no ``AAPL``, no ``MSFT``, no ``QQQ``, no ``BTC``, no anything you
might recognize from the real world. Real symbols WILL be rejected by
the engine and you will lose the bar.

``asset_id`` is CASE-SENSITIVE LOWERCASE. Use ``tier_t1_a01`` exactly,
not ``TIER_T1_A01`` and not ``Tier_T1_A01``. Copy from ``view_universe``
verbatim.

Every user message includes a ``Universe today:`` line listing the
exact asset_id strings legal in the current bar. Use only those.

# 4. TOOLS — exact schemas

Each step you emit ONE JSON object with ``action_type`` and ``payload``.

## 4.1 View tools — never advance the clock

```json
{"action_type": "view_universe",        "payload": {}}
{"action_type": "view_time",            "payload": {}}
{"action_type": "view_portfolio",       "payload": {}}
{"action_type": "view_orders",          "payload": {}}
{"action_type": "view_constraints",     "payload": {}}
{"action_type": "view_episode_metrics", "payload": {}}
```

These return informational ``tool_output`` text. They do NOT trade and
do NOT move time. Use them sparingly — at most one or two per bar.

## 4.2 record_decision — audit log, NOT a trade

```json
{
  "action_type": "record_decision",
  "payload": {
    "regime_label":     "<short label, e.g. 'momentum_long' or 'flat'>",
    "edge_summary":     "<1-2 sentence systematic edge description>",
    "intended_exposure":"<decimal string in [0,1], e.g. '0.50'>",
    "top_convictions":  [
      {"asset_id": "tier_t1_a01", "weight": "0.20"},
      {"asset_id": "tier_t1_a02", "weight": "0.15"}
    ],
    "uncertainty":      "low" | "medium" | "high",
    "reasoning":        "<optional brief systematic rationale>"
  }
}
```

CRITICAL: ``record_decision`` ONLY logs your intent for audit. It does
NOT place any orders, does NOT move money, does NOT change the
portfolio. To actually buy or sell, you MUST call ``place_order``
separately for each asset.

MANDATORY PER-BAR GATE: ``advance_day`` will REJECT with
``decision_required`` if you have not called ``record_decision`` for
the current session_date. Every bar needs a fresh decision — even if
your thesis is unchanged, you must still re-articulate
regime_label/edge_summary/intended_exposure/top_convictions/uncertainty
to confirm you re-evaluated the day's information set.

CALL ``record_decision`` EXACTLY ONCE PER SESSION DATE. Calling it
again on the same date returns ``A decision has already been recorded
for this session date.`` and wastes a step.

The ``edge_summary`` and ``reasoning`` fields are scanned by the
rules-clause verifier. Memorized-history phrases (see § 7) trip a
``-1.0`` reward and may terminate the episode. Stay systematic.

## 4.3 place_order — the actual trade

```json
{
  "action_type": "place_order",
  "payload": {
    "client_order_id": "<unique string, e.g. 'ord_2020-01-02_a01_buy'>",
    "asset_id":        "tier_t1_a01",
    "side":            "buy" | "sell",
    "quantity":        N
  }
}
```

- ``client_order_id`` MUST be unique across the entire episode. Pattern
  ``ord_<date>_<asset>_<side>`` works (date format: ``YYYYMMDD``).
- ``asset_id`` MUST be one of the strings from ``Universe today``.
- ``side`` is ``"buy"`` or ``"sell"`` (lowercase).
- ``quantity`` MUST be an integer ``> 0``. Quantity 0 or negative is
  rejected. Floats are rejected.
- The order queues. It executes at the OPEN of the NEXT session — i.e.
  the next time you call ``advance_day``.

QUANTITY SIZING RECIPE (use this verbatim if you have no better data):

  notional_for_asset = cash_available × weight_of_asset
  quantity           = max(1, floor(notional_for_asset / approx_price))

You can compute ``approx_price`` either by inspecting recent close
in the tool_output of ``view_portfolio`` / ``view_episode_metrics``,
or by running a sandbox_exec snippet (see § 4.6).

## 4.4 cancel_order

```json
{"action_type": "cancel_order", "payload": {"client_order_id": "<existing id>"}}
```

Removes an order from the queue if it has not already filled.

## 4.5 advance_day — REQUIRED to make progress

```json
{"action_type": "advance_day", "payload": {}}
```

Settles all queued orders at the next session open, marks the
portfolio to the new close, emits the bar's reward, increments the
clock by one trading day. CALL THIS EXACTLY ONCE PER BAR. Without it,
nothing happens — you will spin in place until ``max_steps`` is hit
and the episode will end with no trades and no reward.

## 4.6 sandbox_exec — your research desk

```json
{
  "action_type": "sandbox_exec",
  "payload": {
    "command":         ["python", "-c", "<short script>"],
    "env":             {},
    "timeout_seconds": 30
  }
}
```

Runs Python or shell in an isolated worker. Use this to:
- Load the bars parquet and compute returns / vol / Sharpe.
- Fit a small predictive model (sklearn).
- Score a regime classifier you want to drop in.
- Anything else that helps build a systematic edge.

### DATA LAYOUT INSIDE THE SANDBOX (memorize this — wrong paths fail)

The sandbox exposes the catalog as a read-only directory tree. To
build a path that works in BOTH the Docker provider and the local
provider, use the ``WORKSPACE_DATA`` env var injected by the runner:

  os.environ['WORKSPACE_DATA']   ← data dir (the catalog tree)
  os.environ['WORKSPACE_WORK']   ← read-write scratch dir
  os.environ['WORKSPACE_META']   ← metadata / manifest copies
  os.environ['WORKSPACE_OUT']    ← artifacts dir

Inside ``WORKSPACE_DATA`` the layout is EXACTLY:

  daily_bars/part-000.parquet       ← OHLCV, one row per (asset, day)
  asset_master.parquet              ← asset listing/metadata
  calendar.parquet                  ← list of trading session_dates
  corporate_actions.parquet         ← splits, dividends, delistings
  fundamentals_pti.parquet          ← (mostly empty for tier assets)
  episode_manifests/<task>.json     ← your manifest

Build paths via ``os.path.join(os.environ['WORKSPACE_DATA'], 'daily_bars', 'part-000.parquet')``.
DO NOT hardcode ``/workspace/data/...`` — that path only exists inside
Docker. Use the env var or the rollout will fail on the local provider.

Common mistakes the parser sees:
  BAD:  /workspace/data/bars.parquet            (does not exist)
  BAD:  /workspace/data/bars/<asset>.parquet    (no per-asset files)
  BAD:  /workspace/data/bars/*.parquet          (no such directory)
  BAD:  /workspace/data/daily_bars/part-000.parquet  (only works in Docker)
  GOOD: os.path.join(os.environ['WORKSPACE_DATA'], 'daily_bars', 'part-000.parquet')

### BARS SCHEMA

The bars parquet has these columns (one row per asset per session):

  asset_id        str   — the alias, e.g. "tier_t1_a01"
  session_date    date  — trading day (datetime.date)
  open            Decimal
  high            Decimal
  low             Decimal
  close           Decimal
  volume          int64
  dollar_volume   Decimal
  available_at    datetime[UTC]

This is LONG format (NOT pivoted). To get a wide DataFrame keyed by
asset_id, pivot it yourself:

```python
df = pd.read_parquet('/workspace/data/daily_bars/part-000.parquet')
wide = df.pivot(index='session_date', columns='asset_id', values='close')
```

### CONCRETE WORKING SANDBOX_EXEC EXAMPLE

```json
{"action_type":"sandbox_exec","payload":{
  "command":["python","-c","import os, pandas as pd; p=os.path.join(os.environ['WORKSPACE_DATA'],'daily_bars','part-000.parquet'); df=pd.read_parquet(p); df['close']=df['close'].astype(float); wide=df.pivot(index='session_date',columns='asset_id',values='close').sort_index(); rets=wide.pct_change().dropna(); print('last 5 rows:'); print(rets.tail().to_string()); print('mean returns:'); print(rets.mean().to_string())"],
  "timeout_seconds":30
}}
```

This script reads the bars via ``WORKSPACE_DATA``, pivots to
wide-format closes, computes returns, and prints a momentum signal.
Adapt the numerics — the env-var dispatch and pivot pattern are the
right shape and work on both Docker and local providers.

### FORBIDDEN INSIDE THE SANDBOX (auto-detected, -1.0 reward + termination)

``eval``, ``exec``, ``__import__``, ``compile``,
``getattr(_, '__...')``, ``importlib``, ``ctypes``, ``subprocess``,
``socket``, mutation of private ``_cash`` / ``_positions`` attrs.
Stick to ordinary numeric Python.

# 5. PER-BAR PROTOCOL — the only correct loop

Standing still is not a strategy. The clock only moves on advance_day.
For each bar, you should follow this protocol:

  STEP 1.  (Optional, at most ONE call) view_portfolio or view_universe
           or view_episode_metrics if you need fresh state. Skip if
           you already have what you need from the prior step's
           tool_output.

  STEP 2.  (Optional, at most ONE call) sandbox_exec to refresh a
           signal or model — only if you actually expect new data
           to change your decision. Don't recompute identical signals
           every bar.

  STEP 3.  record_decision — MANDATORY exactly ONCE for this session
           date, BEFORE the first place_order AND BEFORE advance_day.
           Pick a non-trivial regime_label and edge_summary describing
           your current systematic view. The env REJECTS advance_day
           with ``decision_required`` if you skip this step. Even when
           your thesis is unchanged, you must still re-record to
           confirm you considered the day's information.

  STEP 4.  place_order — ZERO OR MORE times. Submit one place_order
           per asset you want to buy or sell this bar. If your edge
           is genuinely flat, skip the orders and just advance.

  STEP 5.  advance_day — exactly ONCE. The bar is now over. The next
           observation will be the new bar's data. This call will
           FAIL if step 3 was not done — the per-bar decision gate
           is hard.

You do not have to follow steps 1, 2 every bar — those are optional
inspections. Steps 3 (decision, mandatory), 4 (orders, optional), 5
(advance, mandatory) are the load-bearing sequence each bar.

If you find yourself making 5+ tool calls without an ``advance_day``,
you are LOOPING. Call advance_day immediately.

# 6. WORKED EXAMPLE — one full bar

Suppose the observation is:
```
date=2020-01-15  bars_remaining=45  cash=$100,000.00  positions={}
Universe today: ['tier_t1_a01', 'tier_t1_a02', 'tier_t1_a03',
                 'tier_t1_a04', 'tier_t1_a05']
```

A reasonable bar might look like (one tool call per response):

Response 1 (record your view first):
```json
{"action_type":"record_decision","payload":{
  "regime_label":"diversified_long",
  "edge_summary":"Equal-weight long across the universe — no historical bars yet to size dispersion, so default to 1/N at 60% gross exposure.",
  "intended_exposure":"0.60",
  "top_convictions":[
    {"asset_id":"tier_t1_a01","weight":"0.12"},
    {"asset_id":"tier_t1_a02","weight":"0.12"},
    {"asset_id":"tier_t1_a03","weight":"0.12"},
    {"asset_id":"tier_t1_a04","weight":"0.12"},
    {"asset_id":"tier_t1_a05","weight":"0.12"}
  ],
  "uncertainty":"high"
}}
```

Response 2 (size the first leg — assume approx_price = $75):
quantity = floor(100000 * 0.12 / 75) = 160 shares
```json
{"action_type":"place_order","payload":{
  "client_order_id":"ord_20200115_a01_buy",
  "asset_id":"tier_t1_a01",
  "side":"buy",
  "quantity":160
}}
```

Responses 3–6: same shape, one for each remaining alias, each with
its own unique client_order_id.

Response 7 (settle and advance):
```json
{"action_type":"advance_day","payload":{}}
```

That's a full bar. Total: 1 record_decision + 5 place_order + 1
advance_day = 7 calls. The next observation is day 16.

# 7. FORBIDDEN BEHAVIOR (penalized)

7.1 Real ticker symbols anywhere in payloads. The universe is
    aliased; using ``SPY``, ``AAPL``, ``MSFT``, ``BTC``, etc. will
    not match any asset and the order is rejected.

7.2 Memorized-history claims in ``edge_summary``, ``reasoning``, or
    sandbox stdout. Banned phrases include but are not limited to:
    ``I recall``, ``I remember``, ``in reality``, ``what actually
    happened``, ``the COVID crash``, ``the financial crisis``,
    ``the dot-com bubble``, ``<TICKER> rallied/crashed/surged in
    <YEAR>`` patterns. Trips ``r_rules = -1.0`` and may terminate.

7.3 Sandbox dynamic-evaluation constructs: ``eval(``, ``exec(``,
    ``__import__``, ``compile(``, dunder reflection, ``importlib``,
    ``ctypes``, ``subprocess``, ``socket``, attempts to mutate
    ledger state directly. Trips ``r_hack = -1.0`` and terminates.

7.4 Calling ``record_decision`` more than once per session date.
    The second call is rejected as ``A decision has already been
    recorded`` and burns a step.

7.5 Standing still. Five consecutive non-advance calls means you
    are not making progress. Recover by calling ``advance_day``.

# 8. SELF-VALIDATION (only on the train tier)

There is no separate validation set. The environment exposes a train
tier (T2, 120 bars) and a held-out test tier (T3, 252 bars) — that's
it. If you want to validate a hypothesis before trading on it:

- Pick a sub-window of the past N visible bars (e.g. trailing 40 days).
- Split it: train half + eval half.
- Fit your model on the train half via ``sandbox_exec``.
- Score on the eval half. Only act on the model if eval-score is good.

DO NOT in-sample-evaluate (fitting and scoring on the same span).
That overstates edge and the env will eventually punish you for it.

# 9. OUTPUT FORMAT — strict

Each turn you respond with EXACTLY ONE JSON object inside a single
fenced ``json`` code block. Nothing else — no prose, no commentary,
no extra fences, no second JSON object.

Correct:
```json
{"action_type":"advance_day","payload":{}}
```

If you produce anything other than a single fenced JSON object, the
parser falls back to ``advance_day`` and your bar is wasted.

Begin."""


def _log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={model}", flush=True)


def _log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
    *,
    payload: Optional[dict[str, Any]] = None,
    obs: Any = None,
) -> None:
    err = error if error else "null"
    extras: list[str] = []
    if payload:
        # Compact one-line payload — drop None-valued keys, truncate strings.
        compact = {
            k: (v[:60] + "…" if isinstance(v, str) and len(v) > 60 else v)
            for k, v in payload.items()
            if v is not None
        }
        extras.append(f"payload={json.dumps(compact, separators=(',', ':'))}")
    if obs is not None:
        extras.append(f"value=${obs.portfolio_value:,.2f}")
        if obs.positions:
            extras.append(f"pos={dict(obs.positions)}")
        # Tool output is the only channel for "your order was bad" feedback
        # when obs.error is None — surface it so payload bugs are visible.
        tool_msg = (obs.tool_output or "").strip()
        if tool_msg:
            head = tool_msg.replace("\n", " | ")[:140]
            extras.append(f"msg={head}")
    extra_str = ("  " + "  ".join(extras)) if extras else ""
    print(
        f"[STEP] step={step} action={action} reward={reward:.4f} "
        f"done={str(done).lower()} error={err}{extra_str}",
        flush=True,
    )


def _log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.4f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.4f} rewards={rewards_str}",
        flush=True,
    )


_VALID_ACTION_TYPES = frozenset(
    {
        "view_universe", "view_time", "view_portfolio", "view_orders",
        "view_constraints", "view_episode_metrics", "record_decision",
        "place_order", "cancel_order", "advance_day", "sandbox_exec",
    },
)


def _strip_think_blocks(text: str) -> str:
    """Remove Qwen3-style ``<think>...</think>`` blocks AND any unclosed
    ``<think>`` to end-of-string (handles max_tokens truncation mid-thought).
    """

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Drop a dangling open tag if no closing tag landed (cap reached).
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
    return text


def _balanced_json_objects(text: str) -> list[str]:
    """Return all top-level balanced ``{...}`` substrings in text order.

    Handles nested braces (``top_convictions: [{...}]`` etc.) — the regex
    approach in the previous version would either grab too much (greedy
    ``\\{.*\\}``) or stop at the first inner ``}`` (non-greedy). A small
    bracket-balance scan is more reliable.
    """

    out: list[str] = []
    depth = 0
    start = -1
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    out.append(text[start : i + 1])
                    start = -1
    return out


def _extract_action_json(text: str) -> dict[str, Any]:
    """Parse the LLM response into a TradeAction dict.

    Resolution order:
      1. Strip ``<think>...</think>`` blocks (and unclosed dangling open tag).
      2. Try fenced ``json`` code blocks, last-match-wins.
      3. Fall back to balanced top-level ``{...}`` substrings, last-match-wins.
      4. If nothing parses cleanly, return ``advance_day`` so the simulation
         clock keeps moving — view_portfolio would freeze the rollout in a
         tight loop on garbage outputs.
    """

    text = _strip_think_blocks(text)

    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidates: list[str] = list(fenced)
    if not candidates:
        candidates = _balanced_json_objects(text)

    for candidate in reversed(candidates):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        action = parsed.get("action_type")
        if action in _VALID_ACTION_TYPES:
            return parsed
    return {"action_type": "advance_day", "payload": {}}


def _compact_observation(obs: Any) -> str:
    """Render observation as a compact string for the next LLM prompt."""

    pieces = [
        f"date={obs.current_date}",
        f"phase={obs.phase}",
        f"bars_remaining={obs.bars_remaining}",
        f"value=${obs.portfolio_value:,.2f}",
        f"cash=${obs.cash:,.2f}",
        f"positions={obs.positions}",
        f"last_reward={(obs.reward or 0.0):.4f}",
    ]
    if obs.error:
        pieces.append(f"ERROR={obs.error}")
    if obs.reward_breakdown:
        rb = ",".join(
            f"{k}={v:.4f}"
            for k, v in obs.reward_breakdown.items()
            if v != 0.0
        )
        if rb:
            pieces.append(f"rb=[{rb}]")
    tool_out = (obs.tool_output or "")[:800]
    if tool_out:
        pieces.append("---\n" + tool_out)
    return "\n".join(pieces)


async def run_train_study(
    env: Any,
    client: Any,
    model: str,
    *,
    run_logger: Any = None,
    system_prompt: str | None = None,
) -> str:
    """Run the train tier as a one-shot study phase.

    Loops up to ``TRAIN_STUDY_RETRIES`` LLM calls, sending a corrective
    suffix each time the model fails to emit a ``record_decision``. Falls
    back to a default strategy only after all retries are exhausted.
    Returns the strategy text the agent committed to (concatenation of
    ``edge_summary`` + ``reasoning`` etc.). Empty string on any failure.
    """

    rewards: List[float] = []
    steps_taken = 0
    success = False

    _log_start(task="train", model=model)

    strategy = ""
    attempts_used = 0
    used_fallback = False
    parsed_action: dict[str, Any] = {}

    if run_logger is not None:
        run_logger.begin_phase("train")

    try:
        reset_result = await env.reset(task_tier="train")
        obs = reset_result.observation
        if run_logger is not None:
            run_logger.write_study_packet(obs.tool_output or "")

        base_prompt = (
            f"{obs.tool_output}\n\n"
            "Output ONE record_decision JSON action that summarises the "
            "strategy you commit to for the upcoming TEST rollout. The text "
            "you put into `edge_summary` and `reasoning` is what the test "
            "rollout sees verbatim. Be concrete: regime classification, "
            "sizing rule, rebalance cadence, kill-switch condition. The "
            "JSON object MUST have action_type=\"record_decision\". Do NOT "
            "call view_*, place_order, sandbox_exec, or any other tool here "
            "— only record_decision is accepted in the train tier. JSON only."
        )
        retry_suffix = ""

        active_system_prompt = system_prompt if system_prompt is not None else SYSTEM_PROMPT
        for attempt in range(1, TRAIN_STUDY_RETRIES + 1):
            attempts_used = attempt
            user_prompt = base_prompt + retry_suffix
            try:
                response = await asyncio.to_thread(
                    client.chat.completions.create,
                    model=model,
                    messages=[
                        {"role": "system", "content": active_system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=2048,
                )
                raw = response.choices[0].message.content or ""
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[DEBUG] LLM call failed during train study attempt {attempt}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                raw = ""

            parsed_action = _extract_action_json(raw)
            got_action_type = parsed_action.get("action_type")
            warning = (
                ""
                if got_action_type == "record_decision"
                else f"expected record_decision, got {got_action_type!r}"
            )
            if run_logger is not None:
                run_logger.log_llm_call(
                    step=attempt,
                    prompt_user=user_prompt,
                    raw_response=raw,
                    parsed_action=parsed_action,
                    parse_warning=warning,
                )
            if got_action_type == "record_decision":
                break
            retry_suffix = (
                "\n\nYour previous response did not parse as a valid "
                "record_decision JSON. You MUST output exactly one JSON "
                "object with action_type=\"record_decision\". The train "
                "tier accepts no other tool. Try again, JSON only, no prose."
            )

        if parsed_action.get("action_type") == "record_decision":
            payload = parsed_action.get("payload") or {}
        else:
            used_fallback = True
            payload = {
                "regime_label": "mixed",
                "edge_summary": "Default equal-weight long across the universe.",
                "intended_exposure": "0.8",
                "uncertainty": "high",
                "reasoning": (
                    f"Fallback after {TRAIN_STUDY_RETRIES} attempts — "
                    "model never emitted a parseable record_decision."
                ),
            }
            if run_logger is not None:
                run_logger.add_note(
                    f"train: fallback fired after {TRAIN_STUDY_RETRIES} retries",
                )

        result = await env.step(
            TradeAction(action_type="record_decision", payload=payload),
        )
        steps_taken = 1
        _log_step(
            step=1,
            action="record_decision",
            reward=0.0,
            done=result.done,
            error=result.observation.error,
            payload=payload,
            obs=result.observation,
        )
        if run_logger is not None:
            run_logger.log_step(
                step=1,
                action_type="record_decision",
                payload=payload,
                observation=result.observation,
                reward=0.0,
            )

        edge = str(payload.get("edge_summary") or "").strip()
        reasoning = str(payload.get("reasoning") or "").strip()
        regime = str(payload.get("regime_label") or "").strip()
        exposure = payload.get("intended_exposure")
        uncertainty = str(payload.get("uncertainty") or "").strip()
        parts: list[str] = []
        if regime:
            parts.append(f"Regime: {regime}")
        if exposure is not None:
            parts.append(f"Intended exposure: {exposure}")
        if uncertainty:
            parts.append(f"Uncertainty: {uncertainty}")
        if edge:
            parts.append(f"Edge summary: {edge}")
        if reasoning:
            parts.append(f"Reasoning: {reasoning}")
        strategy = "\n".join(parts)
        success = bool(strategy)

        if run_logger is not None:
            run_logger.write_carry_strategy(strategy)
            run_logger.write_phase_summary(
                {
                    "phase": "train",
                    "attempts_used": attempts_used,
                    "max_attempts": TRAIN_STUDY_RETRIES,
                    "used_fallback": used_fallback,
                    "regime_label": regime,
                    "intended_exposure": exposure,
                    "uncertainty": uncertainty,
                    "edge_summary": edge,
                    "reasoning": reasoning,
                    "carry_strategy_chars": len(strategy),
                    "study_packet_chars": len(obs.tool_output or ""),
                },
            )
            run_logger.record_score("train", 1.0 if success and not used_fallback else 0.0)
    except Exception as exc:  # noqa: BLE001
        print(
            f"[DEBUG] Error during train study: {exc}",
            file=sys.stderr,
            flush=True,
        )
        if run_logger is not None:
            run_logger.add_note(f"train: exception {type(exc).__name__}: {exc}")
    finally:
        _log_end(
            success=success,
            steps=steps_taken,
            score=0.0,
            rewards=rewards,
        )
    return strategy


async def run_task(
    task_id: str,
    env: Any,
    client: Any,
    model: str,
    max_steps: int = _DEFAULT_MAX_STEPS,
    carry_strategy: str = "",
    *,
    run_logger: Any = None,
    system_prompt: str | None = None,
) -> float:
    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.0
    regime_labels: list[str] = []
    cumulative_reward = 0.0
    initial_portfolio_value = 0.0
    final_portfolio_value = 0.0
    bars_completed = 0
    violations_seen: list[str] = []

    _log_start(task=task_id, model=model)

    if run_logger is not None:
        run_logger.begin_phase(task_id)

    try:
        reset_kwargs: dict[str, Any] = {"task_tier": task_id}
        if task_id == "test" and carry_strategy:
            reset_kwargs["carry_strategy"] = carry_strategy
        reset_result = await env.reset(**reset_kwargs)
        obs = reset_result.observation
        initial_portfolio_value = float(obs.portfolio_value or 0.0)

        # Fetch the universe once — it's static for the episode. Inject the
        # exact alias list into every per-step prompt so the model can't
        # hallucinate real ticker symbols (a real failure mode observed
        # with Qwen3 emitting "SPY" / "AAPL" instead of "tier_t1_a01").
        universe_assets: list[str] = []
        try:
            uni_result = await env.step(
                TradeAction(action_type="view_universe", payload={}),
            )
            uni_meta = (uni_result.observation.tool_metadata or {}).get("universe", [])
            universe_assets = [
                row["asset_id"] for row in uni_meta if row.get("asset_id")
            ]
            obs = uni_result.observation
        except Exception as exc:  # noqa: BLE001
            print(
                f"[DEBUG] universe fetch failed: {exc}",
                file=sys.stderr,
                flush=True,
            )

        # Track recent actions so we can nudge the agent when it loops on
        # view_* tools without ever advancing time.
        recent_actions: List[str] = []
        for step_idx in range(1, max_steps + 1):
            if obs.done:
                break
            non_advance_streak = sum(
                1 for a in recent_actions[-6:] if a != "advance_day"
            )
            stuck_hint = ""
            if non_advance_streak >= 5:
                stuck_hint = (
                    "\n\nWARNING: you have made "
                    f"{non_advance_streak} consecutive non-advance calls. "
                    "The clock has not moved. Stop inspecting state and "
                    "either record_decision + place_order or call advance_day "
                    "RIGHT NOW."
                )
            universe_line = (
                f"Universe today (use these exact asset_id strings, "
                f"verbatim, case-sensitive): {universe_assets}\n"
                if universe_assets
                else ""
            )
            prompt = (
                f"{universe_line}"
                f"Current environment state:\n{_compact_observation(obs)}\n\n"
                f"Recent actions (oldest→newest): {recent_actions[-6:] or '[]'}\n"
                f"Step {step_idx} of {max_steps}. "
                f"Choose your next action (JSON only)."
                f"{stuck_hint}"
            )
            active_system_prompt = (
                system_prompt if system_prompt is not None else SYSTEM_PROMPT
            )
            try:
                response = await asyncio.to_thread(
                    client.chat.completions.create,
                    model=model,
                    messages=[
                        {"role": "system", "content": active_system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=2048,
                )
                raw = response.choices[0].message.content or ""
            except Exception as exc:  # noqa: BLE001 - don't crash baseline on API blip
                print(f"[DEBUG] LLM call failed on {task_id} step {step_idx}: {exc}",
                      file=sys.stderr, flush=True)
                raw = ""

            action_dict = _extract_action_json(raw)
            action_type = action_dict.get("action_type", "view_portfolio")
            payload = action_dict.get("payload") or {}
            recent_actions.append(action_type)

            if run_logger is not None:
                run_logger.log_llm_call(
                    step=step_idx,
                    prompt_user=prompt,
                    raw_response=raw,
                    parsed_action={"action_type": action_type, "payload": payload},
                    parse_warning="",
                )

            step_result = await env.step(
                TradeAction(action_type=action_type, payload=payload),
            )
            steps_taken += 1
            reward = float(step_result.reward or 0.0)
            rewards.append(reward)
            cumulative_reward += reward
            tool_meta = step_result.observation.tool_metadata or {}
            tool_err = tool_meta.get("error") if isinstance(tool_meta, dict) else None
            if (
                action_type == "advance_day"
                and not step_result.observation.error
                and not tool_err
            ):
                bars_completed += 1
            if action_type == "record_decision" and isinstance(payload, dict):
                regime_label = str(payload.get("regime_label") or "").strip()
                if regime_label:
                    regime_labels.append(regime_label)
            for v in step_result.observation.violations or []:
                violations_seen.append(str(v))
            _log_step(
                step=steps_taken,
                action=action_type,
                reward=reward,
                done=step_result.done,
                error=step_result.observation.error,
                payload=payload,
                obs=step_result.observation,
            )
            if run_logger is not None:
                run_logger.log_step(
                    step=steps_taken,
                    action_type=action_type,
                    payload=payload,
                    observation=step_result.observation,
                    reward=reward,
                )
            obs = step_result.observation
            if obs.done:
                break

        final_portfolio_value = float(obs.portfolio_value or 0.0)
        score = final_portfolio_value / (
            initial_portfolio_value or 1.0
        ) - 1.0
        success = score > 0.0

    except Exception as exc:  # noqa: BLE001
        print(f"[DEBUG] Error on task '{task_id}': {exc}", file=sys.stderr, flush=True)
        if run_logger is not None:
            run_logger.add_note(f"{task_id}: exception {type(exc).__name__}: {exc}")

    finally:
        _log_end(
            success=success,
            steps=steps_taken,
            score=score,
            rewards=rewards,
        )
        if run_logger is not None:
            score_normalized = (
                max(0.0, min(1.0, cumulative_reward / bars_completed))
                if bars_completed > 0
                else 0.5
            )
            run_logger.write_phase_summary(
                {
                    "phase": task_id,
                    "total_steps": steps_taken,
                    "bars_completed": bars_completed,
                    "max_steps": max_steps,
                    "carry_strategy_used": bool(carry_strategy),
                    "initial_portfolio_value": initial_portfolio_value,
                    "final_portfolio_value": final_portfolio_value,
                    "final_score": score,
                    "cumulative_reward": cumulative_reward,
                    "score_normalized": score_normalized,
                    "regime_labels_seen": regime_labels,
                    "violations": violations_seen,
                },
            )
            run_logger.record_score(task_id, score)

    return score


def _emit_fallback(model: str) -> None:
    """Last-resort output so auto-validators always see three [START]/[END] pairs."""

    for task_id in TASK_IDS:
        _log_start(task=task_id, model=model)
        _log_end(success=False, steps=0, score=0.0, rewards=[0.0])


async def run_baseline_task(
    task_id: str,
    env: Any,
    baseline_name: str,
    max_steps: int = _DEFAULT_MAX_STEPS,
) -> float:
    """Drive a single tier through the named baseline, mirroring run_task's log shape."""

    baseline_fn = BASELINES.get(baseline_name)
    if baseline_fn is None:
        _log_start(task=task_id, model=baseline_name)
        _log_end(success=False, steps=0, score=0.0, rewards=[0.0])
        return 0.0

    _log_start(task=task_id, model=baseline_name)
    score = 0.0
    summary: dict[str, Any] = {}
    try:
        summary = await baseline_fn(env, tier=task_id, max_steps=max_steps)
        for step_idx, (reward, breakdown) in enumerate(
            zip(summary["rewards"], summary["reward_breakdowns"], strict=False),
            start=1,
        ):
            _log_step(
                step=step_idx,
                action="advance_day",
                reward=reward,
                done=(step_idx == len(summary["rewards"])),
                error=None,
            )
        # Score = ROI from initial cash, mirrors run_task's convention.
        # Initial cash is fixed at $100K for every tier per the manifest.
        score = float(summary["final_value"]) / 100_000.0 - 1.0
        success = score > 0.0
    except Exception as exc:  # noqa: BLE001
        print(
            f"[DEBUG] Baseline {baseline_name!r} failed on {task_id}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        success = False

    _log_end(
        success=success,
        steps=summary.get("steps", 0),
        score=score,
        rewards=summary.get("rewards", []),
    )
    return score


async def main_baseline(baseline_name: str) -> None:
    """Drive every tier through ``baseline_name`` against a running env."""

    if TradeBenchEnv is None or TradeAction is None:
        print(
            "[DEBUG] Missing tradebench client / openenv-core",
            file=sys.stderr,
            flush=True,
        )
        _emit_fallback(baseline_name)
        return

    image_name = os.environ.get("LOCAL_IMAGE_NAME")
    env: Any = None
    scores: dict[str, float] = {}
    try:
        if image_name:
            env = await TradeBenchEnv.from_docker_image(image_name)
        else:
            env_url = os.environ.get("ENV_URL", HF_SPACE_URL)
            env = TradeBenchEnv(
                base_url=env_url,
                connect_timeout_s=120,
                message_timeout_s=300,
            )
        async with env:
            for task_id in TASK_IDS:
                scores[task_id] = await run_baseline_task(task_id, env, baseline_name)
    except Exception as exc:  # noqa: BLE001
        print(f"[DEBUG] Baseline top-level error: {exc}", file=sys.stderr, flush=True)

    finally:
        for task_id in TASK_IDS:
            if task_id not in scores:
                _log_start(task=task_id, model=baseline_name)
                _log_end(success=False, steps=0, score=0.0, rewards=[0.0])
                scores[task_id] = 0.0
        if env is not None:
            try:
                await env.close()
            except Exception:  # noqa: BLE001
                pass

    print(f"\n{'=' * 60}", flush=True)
    print(f"BASELINE SCORES — {baseline_name}", flush=True)
    print(f"{'=' * 60}", flush=True)
    for task_id, s in scores.items():
        print(f"  {task_id:>4}: {s:+.4f}", flush=True)


async def main(
    *,
    system_prompt: str | None = None,
    run_id_suffix: str | None = None,
) -> dict[str, Any]:
    """Run one full rollout (train + test or whatever TASK_IDS contains).

    Parameters
    ----------
    system_prompt:
        If provided, overrides the module-level ``SYSTEM_PROMPT`` for both
        the train study phase and the test rollout. Used by the reflection
        driver to swap in an iteratively-improved prompt.
    run_id_suffix:
        Optional tag appended to the auto-generated run id; useful for
        labeling reflection iterations (e.g. ``"iter2"``).

    Returns
    -------
    A dict with keys ``run_id``, ``artifacts_root``, ``scores`` (per-tier
    score dict), ``test_score``, ``test_cumulative_reward``,
    ``test_bars_completed``. Empty/zero values when no rollout was actually
    executed (e.g. missing client libraries).
    """
    hf_token = os.environ.get("HF_TOKEN")
    api_base_url = os.environ.get("API_BASE_URL")
    if api_base_url is None:
        api_base_url = (
            "https://router.huggingface.co/v1"
            if hf_token
            else "https://api.openai.com/v1"
        )
    is_hf_route = "router.huggingface.co" in api_base_url
    default_model = "Qwen/Qwen3-32B:groq" if is_hf_route else "gpt-4o-mini"
    model_name = os.environ.get("MODEL_NAME", default_model)
    api_key = os.environ.get("OPENAI_API_KEY") or hf_token
    image_name = os.environ.get("LOCAL_IMAGE_NAME")
    print(
        f"[CONFIG] base_url={api_base_url}  model={model_name}  "
        f"route={'hf' if is_hf_route else 'openai'}",
        flush=True,
    )

    if OpenAI is None or TradeBenchEnv is None or TradeAction is None:
        missing = []
        if OpenAI is None:
            missing.append("openai")
        if TradeBenchEnv is None or TradeAction is None:
            missing.append("openenv-core / tradebench client")
        print(
            f"[DEBUG] Missing packages: {', '.join(missing)}",
            file=sys.stderr,
            flush=True,
        )
        _emit_fallback(model_name)
        return {
            "run_id": None,
            "artifacts_root": None,
            "scores": {tid: 0.0 for tid in TASK_IDS},
            "test_score": 0.0,
            "test_cumulative_reward": 0.0,
            "test_bars_completed": 0,
            "test_score_normalized": 0.5,
            "test_action_counts": {},
        }

    active_system_prompt = system_prompt if system_prompt is not None else SYSTEM_PROMPT
    client = OpenAI(base_url=api_base_url, api_key=api_key or "dummy")

    scores: dict[str, float] = {}
    env: Any = None
    run_logger: Any = None
    run_id: str | None = None
    artifacts_root: Any = None
    if RunLogger is not None and make_run_id is not None:
        from pathlib import Path as _Path
        tier_pair = "_".join(TASK_IDS)
        run_id = make_run_id(model_name, tier_pair)
        if run_id_suffix:
            run_id = f"{run_id}__{run_id_suffix}"
        run_logger = RunLogger(
            artifacts_root=_Path(ARTIFACTS_ROOT),
            run_id=run_id,
            model=model_name,
            tier_pair=tier_pair,
        )
        run_logger.write_system_prompt(active_system_prompt)
        artifacts_root = run_logger.root
        print(f"[CONFIG] run_id={run_id}", flush=True)
        print(f"[CONFIG] artifacts={run_logger.root}", flush=True)

    try:
        if image_name:
            env = await TradeBenchEnv.from_docker_image(image_name)
        else:
            env_url = os.environ.get("ENV_URL", HF_SPACE_URL)
            env = TradeBenchEnv(
                base_url=env_url,
                connect_timeout_s=120,
                message_timeout_s=300,
            )
        async with env:
            carry_strategy = ""
            for task_id in TASK_IDS:
                if task_id == "train":
                    carry_strategy = await run_train_study(
                        env,
                        client,
                        model_name,
                        run_logger=run_logger,
                        system_prompt=active_system_prompt,
                    )
                    scores[task_id] = 1.0 if carry_strategy else 0.0
                else:
                    scores[task_id] = await run_task(
                        task_id,
                        env,
                        client,
                        model_name,
                        carry_strategy=carry_strategy if task_id == "test" else "",
                        run_logger=run_logger,
                        system_prompt=active_system_prompt,
                    )
    except Exception as exc:  # noqa: BLE001
        print(f"[DEBUG] Top-level error: {exc}", file=sys.stderr, flush=True)
        if run_logger is not None:
            run_logger.add_note(f"top-level exception: {type(exc).__name__}: {exc}")

    finally:
        for task_id in TASK_IDS:
            if task_id not in scores:
                _log_start(task=task_id, model=model_name)
                _log_end(success=False, steps=0, score=0.0, rewards=[0.0])
                scores[task_id] = 0.0
        if env is not None:
            try:
                await env.close()
            except Exception:  # noqa: BLE001
                pass
        if run_logger is not None:
            run_logger.close()

    print(f"\n{'=' * 60}", flush=True)
    print("FINAL SCORES", flush=True)
    print(f"{'=' * 60}", flush=True)
    for task_id, s in scores.items():
        print(f"  {task_id:>4}: {s:+.4f}", flush=True)
    avg = sum(scores.values()) / len(scores) if scores else 0.0
    print(f"  avg : {avg:+.4f}", flush=True)
    if run_logger is not None:
        print(f"  run : {run_logger.root}", flush=True)

    test_summary: dict[str, Any] = {}
    if artifacts_root is not None:
        test_summary_path = artifacts_root / "test" / "summary.json"
        if test_summary_path.is_file():
            try:
                test_summary = json.loads(test_summary_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                test_summary = {}

    test_cumulative_reward = float(test_summary.get("cumulative_reward", 0.0))
    test_bars = int(test_summary.get("bars_completed", 0))
    if test_bars > 0:
        test_score_normalized = max(0.0, min(1.0, test_cumulative_reward / test_bars))
    else:
        test_score_normalized = 0.5
    test_action_counts: dict[str, int] = {}
    if artifacts_root is not None:
        manifest_path = artifacts_root / "manifest.json"
        if manifest_path.is_file():
            try:
                manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
                test_action_counts = dict(
                    manifest_data.get("action_counts", {}).get("test", {}) or {},
                )
            except Exception:  # noqa: BLE001
                test_action_counts = {}

    return {
        "run_id": run_id,
        "artifacts_root": str(artifacts_root) if artifacts_root is not None else None,
        "scores": scores,
        "test_score": float(scores.get("test", 0.0)),
        "test_cumulative_reward": test_cumulative_reward,
        "test_bars_completed": test_bars,
        "test_score_normalized": test_score_normalized,
        "test_action_counts": test_action_counts,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        choices=sorted(BASELINES.keys()),
        default=None,
        help=(
            "Run a deterministic baseline (cash, equal_weight, kelly) instead of "
            "the LLM driver. Emits the same [START]/[STEP]/[END] log shape so the "
            "auto-validator parses identically."
        ),
    )
    return parser


if __name__ == "__main__":
    try:
        args = _build_arg_parser().parse_args()
        if args.baseline is not None:
            asyncio.run(main_baseline(args.baseline))
        else:
            asyncio.run(main())
    except Exception as exc:  # noqa: BLE001
        print(f"[DEBUG] Fatal error: {exc}", file=sys.stderr, flush=True)
        _emit_fallback(os.environ.get("MODEL_NAME", "unknown"))
