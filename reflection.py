"""GEPA-style reflective prompt optimisation for the TradeBench rollout.

A reflection driver passes the current SYSTEM_PROMPT plus a curated trace
of one rollout's trajectory and reward breakdown to a stronger model
(Anthropic Claude Opus 4.7 via OpenRouter) and asks it to return an
improved system prompt. The returned prompt drives the next rollout.
After N iterations we plot reward vs iteration to show prompt-optimisation
progress.

GEPA reference: https://github.com/gepa-ai/gepa. The meta-prompt below
adapts the GEPA reflection structure to our setting where data is
revealed bar-by-bar (not all at once) and the score is a multi-component
composite reward.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - dev fallback
    OpenAI = None  # type: ignore[assignment,misc]

DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_REFLECTION_MODEL = os.environ.get(
    "REFLECTION_MODEL",
    "anthropic/claude-opus-4.7",
)
DEFAULT_REFLECTION_MAX_TOKENS = int(
    os.environ.get("REFLECTION_MAX_TOKENS", "16384"),
)


@dataclass(frozen=True)
class ReflectionContext:
    """All inputs the reflector needs to propose a better system prompt."""

    iteration: int
    current_system_prompt: str
    study_packet: str
    carry_strategy: str
    train_summary: dict[str, Any]
    test_summary: dict[str, Any]
    trajectory_excerpts: str
    llm_call_excerpts: str
    failure_signals: str
    history_snapshot: list[dict[str, Any]] = ()  # type: ignore[assignment]
    prior_prompt_chars: int = 0
    target_prompt_chars: int = 0


def make_openrouter_client(api_key: str | None = None) -> Any:
    """Return an OpenAI-compatible client pointed at OpenRouter."""

    if OpenAI is None:
        msg = "openai package is not installed; cannot create OpenRouter client"
        raise RuntimeError(msg)
    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        msg = (
            "OpenRouter API key missing. Set OPENROUTER_API_KEY or pass "
            "api_key to make_openrouter_client()."
        )
        raise RuntimeError(msg)
    return OpenAI(
        base_url=DEFAULT_OPENROUTER_BASE_URL,
        api_key=key,
        default_headers={
            "HTTP-Referer": "https://github.com/yobro4619/tradebench",
            "X-Title": "TradeBench reflection driver",
        },
    )


def load_run_context(
    run_root: Path,
    *,
    iteration: int,
    max_traj_bars: int = 20,
    max_excerpt_calls: int = 6,
    history_snapshot: list[dict[str, Any]] | None = None,
) -> ReflectionContext:
    """Read one rollout's artifacts and build a ``ReflectionContext``.

    ``history_snapshot`` is the list of prior history rows (baseline +
    reflections so far). When supplied, the meta-prompt will render a
    comparison table so the reflector can see whether its prior edit
    helped or hurt versus the baseline. Each row is a dict with keys
    ``iteration``, ``test_score``, ``test_cumulative_reward``,
    ``test_score_normalized``, ``test_action_counts`` (subset).
    """

    run_root = Path(run_root).resolve()
    system_prompt = (run_root / "system_prompt.txt").read_text(encoding="utf-8")
    study_packet = _read_optional(run_root / "train" / "study_packet.txt")
    carry_strategy = _read_optional(run_root / "carry_strategy.txt")
    train_summary = _read_optional_json(run_root / "train" / "summary.json")
    test_summary = _read_optional_json(run_root / "test" / "summary.json")

    trajectory_path = run_root / "test" / "trajectory.jsonl"
    llm_calls_path = run_root / "test" / "llm_calls.jsonl"
    trajectory = _read_jsonl(trajectory_path)
    llm_calls = _read_jsonl(llm_calls_path)

    trajectory_excerpts = _render_trajectory_excerpts(
        trajectory,
        max_bars=max_traj_bars,
    )
    llm_call_excerpts = _render_llm_call_excerpts(
        trajectory=trajectory,
        llm_calls=llm_calls,
        max_excerpts=max_excerpt_calls,
    )
    failure_signals = _summarise_failure_signals(trajectory, llm_calls, test_summary)

    prior_chars = len(system_prompt)
    target_chars = int(prior_chars * 1.10)

    return ReflectionContext(
        iteration=iteration,
        current_system_prompt=system_prompt,
        study_packet=study_packet,
        carry_strategy=carry_strategy,
        train_summary=train_summary,
        test_summary=test_summary,
        trajectory_excerpts=trajectory_excerpts,
        llm_call_excerpts=llm_call_excerpts,
        failure_signals=failure_signals,
        history_snapshot=list(history_snapshot or []),
        prior_prompt_chars=prior_chars,
        target_prompt_chars=target_chars,
    )


def build_reflection_meta_prompt(ctx: ReflectionContext) -> str:
    """Compose the user-message text sent to the reflector model."""

    parts: list[str] = []
    parts.append(_META_PROMPT_HEADER.strip())
    parts.append("")
    parts.append(f"## Reflection iteration: {ctx.iteration}")
    parts.append("")
    parts.append("## Length budget")
    parts.append(
        f"Prior prompt: {ctx.prior_prompt_chars} chars. "
        f"Maximum allowed for your rewrite: {ctx.target_prompt_chars} chars (1.10x). "
        f"Aim for the same length; never exceed the cap.",
    )
    parts.append("")
    if ctx.history_snapshot:
        parts.append("## Comparison vs baseline and prior iterations")
        parts.append(_render_history_table(ctx.history_snapshot))
        parts.append("")
    parts.append("## Most recent rollout - score and reward analysis")
    parts.append(_render_score_block(ctx))
    parts.append("")
    parts.append("## Failure signals (auto-derived from trace)")
    parts.append(ctx.failure_signals or "(no specific failure signals detected)")
    parts.append("")
    parts.append("## Trajectory excerpts (curated, bar-by-bar)")
    parts.append("")
    parts.append(ctx.trajectory_excerpts or "(trajectory empty)")
    parts.append("")
    parts.append("## Representative model reasoning excerpts")
    parts.append("")
    parts.append(ctx.llm_call_excerpts or "(no excerpts)")
    parts.append("")
    if ctx.carry_strategy:
        parts.append("## Strategy committed during the train study")
        parts.append("```")
        parts.append(ctx.carry_strategy)
        parts.append("```")
        parts.append("")
    if ctx.study_packet:
        parts.append("## Train study packet (the data the agent saw before test)")
        parts.append("```")
        parts.append(_truncate(ctx.study_packet, 4000))
        parts.append("```")
        parts.append("")
    parts.append("## Current system prompt (the prompt to improve)")
    parts.append("```")
    parts.append(ctx.current_system_prompt)
    parts.append("```")
    parts.append("")
    parts.append(_META_PROMPT_TASK.strip())
    return "\n".join(parts)


def _render_history_table(history: list[dict[str, Any]]) -> str:
    """Render a simple text table summarising prior runs.

    Each row should have ``iteration``, ``test_score``,
    ``test_cumulative_reward``, ``test_score_normalized``,
    and ``test_action_counts`` (a dict of action -> count).
    """

    if not history:
        return "(no history yet)"
    headers = (
        "iter", "score_norm", "cum_reward", "test_roi",
        "place_orders", "advance_day", "record_decision", "sandbox_exec",
    )
    rows: list[tuple[str, ...]] = []
    for row in history:
        counts = row.get("test_action_counts", {}) or {}
        label = "baseline" if row.get("iteration", 0) == 0 else f"iter_{int(row['iteration']):02d}"
        rows.append(
            (
                label,
                f"{float(row.get('test_score_normalized', 0.5)):.3f}",
                f"{float(row.get('test_cumulative_reward', 0.0)):+.4f}",
                f"{float(row.get('test_score', 0.0)):+.4f}",
                str(int(counts.get("place_order", 0))),
                str(int(counts.get("advance_day", 0))),
                str(int(counts.get("record_decision", 0))),
                str(int(counts.get("sandbox_exec", 0))),
            ),
        )
    widths = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    line_fmt = "  " + "  ".join(f"{{:<{w}}}" for w in widths)
    out = [line_fmt.format(*headers), line_fmt.format(*("-" * w for w in widths))]
    for r in rows:
        out.append(line_fmt.format(*r))
    if len(history) >= 2:
        prev = history[-2]
        cur = history[-1]
        prev_norm = float(prev.get("test_score_normalized", 0.5))
        cur_norm = float(cur.get("test_score_normalized", 0.5))
        prev_orders = int((prev.get("test_action_counts") or {}).get("place_order", 0))
        cur_orders = int((cur.get("test_action_counts") or {}).get("place_order", 0))
        delta_norm = cur_norm - prev_norm
        delta_orders = cur_orders - prev_orders
        verdict = "REGRESSED" if delta_norm < -0.01 else (
            "IMPROVED" if delta_norm > 0.01 else "NEUTRAL"
        )
        out.append("")
        out.append(
            f"  delta vs prior: score_norm={delta_norm:+.3f}, "
            f"place_orders={delta_orders:+d} -> {verdict}",
        )
        if verdict == "REGRESSED":
            out.append(
                "  IMPORTANT: your prior edit hurt the metric. The current "
                "edit MUST partially or fully reverse the change that caused "
                "the regression. Examine what your prior edit added or removed.",
            )
    return "\n".join(out)


def propose_improved_prompt(
    ctx: ReflectionContext,
    *,
    client: Any,
    model: str = DEFAULT_REFLECTION_MODEL,
    max_tokens: int = DEFAULT_REFLECTION_MAX_TOKENS,
) -> tuple[str, str, str, str]:
    """Call the reflector LLM and return reflection artifacts.

    Returns a four-tuple ``(new_system_prompt, raw_response,
    reflector_system_prompt, meta_prompt)`` so the caller can persist
    every input and output verbatim. Extraction strategy: prefer the
    text between ``<NEW_SYSTEM_PROMPT>`` and ``</NEW_SYSTEM_PROMPT>``,
    fall back to the first fenced block, fall back to the raw response.
    """

    reflector_system_prompt = _REFLECTOR_SYSTEM_PROMPT.strip()
    meta_prompt = build_reflection_meta_prompt(ctx)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": reflector_system_prompt},
            {"role": "user", "content": meta_prompt},
        ],
        max_tokens=max_tokens,
    )
    raw = response.choices[0].message.content or ""
    new_prompt = _extract_new_prompt(raw)
    return new_prompt, raw, reflector_system_prompt, meta_prompt


def _extract_new_prompt(raw: str) -> str:
    """Pull the new system prompt out of the reflector's response."""

    match = re.search(
        r"<NEW_SYSTEM_PROMPT>\s*(.*?)\s*</NEW_SYSTEM_PROMPT>",
        raw,
        flags=re.DOTALL,
    )
    if match:
        return match.group(1).strip()
    fenced = re.search(r"```(?:[a-zA-Z]*\n)?(.*?)```", raw, flags=re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    return raw.strip()


def _read_optional(path: Path) -> str:
    if not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _render_trajectory_excerpts(
    trajectory: list[dict[str, Any]],
    *,
    max_bars: int,
) -> str:
    if not trajectory:
        return ""
    lines = [
        "Format: step | action | reward | date | bars_left | pv | positions | error",
    ]
    take = trajectory[:max_bars]
    for r in take:
        pos = r.get("positions") or {}
        pos_str = ", ".join(f"{k}={v:.0f}" for k, v in sorted(pos.items())) or "{}"
        err = r.get("error") or ""
        lines.append(
            f"  step={r.get('step', '?'):>3} | "
            f"{str(r.get('action_type','?')):14s} | "
            f"r={float(r.get('reward', 0.0)):+.4f} | "
            f"{r.get('current_date', '?')} | "
            f"left={int(r.get('bars_remaining', 0)):3d} | "
            f"pv={float(r.get('portfolio_value', 0.0)):.2f} | "
            f"pos={pos_str} | "
            f"err={err}",
        )
    if len(trajectory) > max_bars:
        omitted = len(trajectory) - max_bars
        lines.append(f"  ... [{omitted} more steps omitted]")
    last = trajectory[-1]
    lines.append(
        "  FINAL: pv={pv:.2f} pos={pos} done={done}".format(
            pv=float(last.get("portfolio_value", 0.0)),
            pos=last.get("positions") or {},
            done=last.get("done"),
        ),
    )
    return "\n".join(lines)


def _render_llm_call_excerpts(
    *,
    trajectory: list[dict[str, Any]],
    llm_calls: list[dict[str, Any]],
    max_excerpts: int,
) -> str:
    if not llm_calls:
        return ""
    by_step = {r.get("step"): r for r in trajectory}
    scored: list[tuple[float, dict[str, Any]]] = []
    for call in llm_calls:
        step = call.get("step")
        traj = by_step.get(step) or {}
        reward = float(traj.get("reward", 0.0))
        scored.append((reward, call))
    scored.sort(key=lambda pair: pair[0])
    worst = scored[: max(1, max_excerpts // 2)]
    best = scored[-max(1, max_excerpts - len(worst)) :]
    seen_steps: set[int] = set()
    chosen: list[tuple[str, dict[str, Any]]] = []
    for reward, call in worst:
        step = call.get("step")
        if step in seen_steps:
            continue
        seen_steps.add(step)
        chosen.append(("worst", call))
    for reward, call in reversed(best):
        step = call.get("step")
        if step in seen_steps:
            continue
        seen_steps.add(step)
        chosen.append(("best", call))
    chosen = chosen[:max_excerpts]

    lines: list[str] = []
    for tag, call in chosen:
        step = call.get("step")
        traj = by_step.get(step) or {}
        reward = traj.get("reward", 0.0)
        action = traj.get("action_type") or call.get("parsed_action", {}).get("action_type")
        raw = call.get("raw_response", "")
        reasoning = _extract_thinking(raw) or _truncate(raw, 600)
        lines.append(
            f"### {tag.upper()} step={step} action={action} reward={float(reward):+.4f}",
        )
        parsed = call.get("parsed_action") or {}
        lines.append(f"parsed_action={json.dumps(parsed, ensure_ascii=False)[:400]}")
        if reasoning:
            lines.append("reasoning_excerpt:")
            lines.append(_indent(_truncate(reasoning, 800), "  "))
        if call.get("parse_warning"):
            lines.append(f"parse_warning: {call['parse_warning']}")
        lines.append("")
    return "\n".join(lines)


def _summarise_failure_signals(
    trajectory: list[dict[str, Any]],
    llm_calls: list[dict[str, Any]],
    test_summary: dict[str, Any],
) -> str:
    if not trajectory:
        return ""
    action_counts: dict[str, int] = {}
    error_counts: dict[str, int] = {}
    rejection_msgs: dict[str, int] = {}
    component_totals: dict[str, float] = {}
    for r in trajectory:
        a = str(r.get("action_type", "?"))
        action_counts[a] = action_counts.get(a, 0) + 1
        err = r.get("error")
        if err:
            error_counts[str(err)] = error_counts.get(str(err), 0) + 1
        for comp, val in (r.get("reward_breakdown") or {}).items():
            try:
                component_totals[comp] = component_totals.get(comp, 0.0) + float(val)
            except (TypeError, ValueError):
                continue
        tool_excerpt = (r.get("tool_output_excerpt") or "")[:200]
        for marker in (
            "decision has already been recorded",
            "Order rejected",
            "Look-ahead violation",
            "rules-clause violation",
        ):
            if marker.lower() in tool_excerpt.lower():
                rejection_msgs[marker] = rejection_msgs.get(marker, 0) + 1

    lines = []
    lines.append(f"action_counts: {action_counts}")
    if error_counts:
        lines.append(f"error_counts: {error_counts}")
    if rejection_msgs:
        lines.append(f"env_rejections: {rejection_msgs}")
    lines.append(
        "reward_components_cumulative: "
        + ", ".join(f"{k}={v:+.4f}" for k, v in sorted(component_totals.items())),
    )
    bars = int(test_summary.get("bars_completed", 0))
    place = action_counts.get("place_order", 0)
    if bars > 0:
        ratio = place / bars
        lines.append(
            f"orders_per_bar: {ratio:.2f} (places/bar; under 0.3 typically means undertrading)",
        )
    if action_counts.get("sandbox_exec", 0) > action_counts.get("place_order", 0) * 4:
        lines.append(
            "  signal: heavy sandbox_exec usage relative to place_order — agent "
            "may be over-analysing instead of trading.",
        )
    if rejection_msgs.get("decision has already been recorded", 0) > 3:
        lines.append(
            "  signal: many duplicate record_decision calls per session — agent "
            "is mis-counting decisions per bar.",
        )
    return "\n".join(lines)


def _render_score_block(ctx: ReflectionContext) -> str:
    parts: list[str] = []
    parts.append("```")
    parts.append(json.dumps(ctx.test_summary, indent=2, ensure_ascii=False))
    parts.append("```")
    return "\n".join(parts)


def _extract_thinking(raw: str) -> str:
    match = re.search(r"<think>(.*?)</think>", raw, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"... [truncated {len(text) - limit} chars]"


def _indent(text: str, prefix: str) -> str:
    return "\n".join(prefix + line for line in text.split("\n"))


_REFLECTOR_SYSTEM_PROMPT = """
You are a senior prompt engineer performing reflective optimisation on the
SYSTEM PROMPT of an LLM-driven trading agent in the TradeBench environment.
You are NOT the trading agent. You produce ONE rewritten prompt per call.

# What the agent actually does

The agent runs two phases:

  TRAIN (one LLM call): receives a 252-day study packet in-context with
  per-asset summary stats and a correlation matrix. Must emit a single
  record_decision summarising the strategy it commits to.

  TEST (held-out 120-day rollout, one LLM call per step): observes one
  bar at a time. The full daily_bars parquet is on disk inside the
  sandbox, but the env enforces a point-in-time gate (session_date <=
  current_date). Each bar the agent should follow this protocol:
    1. Optionally call view_* tools to inspect state.
    2. Optionally call sandbox_exec to derive a signal from past bars.
    3. Call record_decision EXACTLY ONCE per session date.
    4. Call place_order zero or more times to push toward target weights.
    5. Call advance_day EXACTLY ONCE to roll the clock to the next bar.

The reward is a seven-component composite per bar: r_wealth (unbounded
log-wealth) plus six bounded regularisers (sharpe_bonus, drawdown,
turnover, concentration, rules, hack), with secondary terms weighted 0.5x.
Cumulative reward over the test rollout is what we are trying to lift.

# The single hardest lesson from prior reflection iterations

Aggressive rewrites BREAK this agent. Specifically the following changes
have caused catastrophic regressions in prior iterations:

  - Adding rigid quotas like "place 3-7 orders per bar" or "PLACE ORDERS
    EVERY BAR" caused the agent to freeze with ZERO orders for 80 steps.
    The agent reads the contradiction between "you must trade" and "you
    must size correctly" and resolves it by emitting advance_day instead.

  - Lengthening the prompt by more than ~10% degraded JSON-action
    selection because it ate the agent's reasoning headroom.

  - Replacing or paraphrasing the §6 WORKED EXAMPLE (the one-bar dialog)
    is dangerous; that example is the agent's primary teacher. Tweaking
    its wording is OK, deleting or rewriting its structure is not.

  - Adding new top-level rules ABOVE §1 confused the agent about which
    section was authoritative.

# Your operating constraints (non-negotiable)

  1. SURGICAL EDITS ONLY. The smallest change that addresses the largest
     diagnosed weakness wins. Prefer to amend ONE phrase, soften ONE
     wording, or add ONE caution. Do not restructure.

  2. LENGTH BUDGET: the new prompt MUST be at most 1.10x the prior
     prompt's length in characters. Aim for roughly the same length;
     drop redundant text before exceeding the cap. Examples are the
     agent's primary teacher and must NEVER be deleted or shortened.

  3. NO RIGID QUOTAS. Never write "place X orders per bar", "issue Y
     trades per session", "always do Z". Trading is regime-dependent.
     Replace any such temptation with a CONDITIONAL — "place orders
     proportional to the gap between target and current weights; if
     no significant gap, advance the clock."

  4. PRESERVE the strict JSON output contract, the 11-tool surface
     (view_universe, view_time, view_portfolio, view_orders,
     view_constraints, view_episode_metrics, record_decision,
     place_order, cancel_order, advance_day, sandbox_exec), the
     §-numbered structure (§1 GAME, §2 WORLD MODEL, §3 UNIVERSE, §4
     TOOLS, §5 PER-BAR PROTOCOL, §6 WORKED EXAMPLE, §7 FORBIDDEN, §8
     SELF-VALIDATION, §9 OUTPUT FORMAT), and the alias-only universe
     rule (asset_id strings must match the Universe today line).

  5. NO new top-level sections above §1. Edit content within sections.

  6. Avoid emojis and non-keyboard symbols. Plain ASCII + standard
     punctuation only. No bold/italic markdown that contradicts the
     JSON-only output contract.

# How to reason

Diagnose the SINGLE most-impactful failure pattern in the trace. State
strengths the prompt already produces correctly. Then propose the
smallest edit that moves the failure pattern without breaking strengths.
You will be shown a comparison table of action counts vs the baseline
and prior iterations - if your prior edit caused a regression, your
current edit MUST partially or fully reverse the change that hurt.

# Output contract

Wrap your final, complete rewritten system prompt between literal tags
``<NEW_SYSTEM_PROMPT>`` and ``</NEW_SYSTEM_PROMPT>``. Before the opening
tag put your one-sentence diagnosis and your three-bullet list of
strengths-to-preserve. Nothing after the closing tag. The driver
extracts text between the tags verbatim.
""".strip()


_META_PROMPT_HEADER = """
You will receive: (1) the current system prompt driving the agent, (2) a
curated trace of the most recent test rollout (per-bar trajectory + the
agent's reasoning excerpts on its best and worst bars), (3) the final
score breakdown, (4) failure signals derived from the trace, and (5) the
strategy and study packet the agent saw before the rollout. Use these to
diagnose the agent's most-impactful weakness and rewrite the system prompt
to address it.
""".strip()


_META_PROMPT_TASK = """
## Your task

Output, in this exact order:

  STEP 0 - STRENGTHS (3 bullets, one line each).
  Identify three SPECIFIC behaviors the current prompt produces correctly,
  citing the trace. Examples of valid bullets:
  - "agent uses os.environ['WORKSPACE_DATA'] consistently in sandbox_exec
    (saw on every sandbox call)"
  - "agent emits valid record_decision JSON with regime_label and
    edge_summary every bar it records"
  - "agent only references aliases tier_a01..tier_a10, no real tickers"
  Your final edit MUST NOT remove, contradict, or shorten the prompt
  sections that produce these strengths.

  STEP 1 - DIAGNOSIS (one sentence).
  The single most-impactful failure pattern from the trace and failure
  signals. Prefer behavior over policy ("agent never re-balances after
  the first set of fills") over abstractions ("agent under-trades").
  If a comparison table is present and your prior edit regressed any
  metric, the diagnosis MUST acknowledge the regression and your edit
  MUST partially or fully reverse the prior change.

  STEP 2 - SURGICAL PLAN (3-6 bullets).
  Each bullet names the EXACT section being edited (e.g. "§5 PER-BAR
  PROTOCOL step 4") and what changes (e.g. "soften 'optional' to 'place
  orders if any target weight differs from current by >2%'"). No new
  top-level sections. No rigid quotas. Total length budget: new prompt
  must be at most 1.10x the prior prompt's character count. If your plan
  exceeds the budget, drop the lowest-priority bullet first.

  STEP 3 - <NEW_SYSTEM_PROMPT>...</NEW_SYSTEM_PROMPT>
  The complete rewritten system prompt between the literal tags. Same
  §-numbered structure, same tool surface, same JSON-only output
  contract, same alias rule. Plain ASCII only. Nothing after the
  closing tag.

# Forbidden patterns (these have all caused regressions in prior runs)

  - Adding "PLACE ORDERS EVERY BAR" / "issue 3-7 orders per bar" or any
    fixed-count quota. Use conditional triggers tied to weight gaps.
  - Adding a new "MOST CRITICAL RULE" or "TOP PRIORITY" section above
    §1. Edit within the existing structure.
  - Lengthening the prompt by more than 10%. Aim for the same length.
  - Deleting or paraphrasing the §6 WORKED EXAMPLE one-bar dialogue.
    You may add ONE small caution sentence inside it; you may not
    rewrite its structure.
  - Replacing concrete examples ("BUY 100 shares of tier_a01") with
    abstract guidance ("size positions appropriately"). Examples teach;
    abstractions cause paralysis.
""".strip()


__all__ = [
    "ReflectionContext",
    "build_reflection_meta_prompt",
    "load_run_context",
    "make_openrouter_client",
    "propose_improved_prompt",
    "DEFAULT_OPENROUTER_BASE_URL",
    "DEFAULT_REFLECTION_MODEL",
]
