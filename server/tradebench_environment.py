"""OpenEnv ``Environment`` subclass wrapping ``TradeBenchSessionRuntime``.

Bridges the OpenEnv wire protocol (``reset`` / ``step`` / ``state``
returning typed Pydantic messages) to the TradeBench internals (tool
dispatch, event-sourced ledger, PIT data service, Docker sandbox).
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

try:
    from ..models import TradeAction, TradeObservation, TradeState
except (ImportError, ModuleNotFoundError):
    from models import TradeAction, TradeObservation, TradeState  # type: ignore[no-redef]

from tradebench.environment.session import TradeBenchSessionRuntime
from tradebench.episodes.study_packet import build_study_packet
from tradebench.settings import resolve_settings
from tradebench.environment.tools import (
    AdvanceDayInput,
    CancelOrderInput,
    PlaceOrderInput,
    RecordDecisionInput,
    SandboxExecInput,
    ViewConstraintsInput,
    ViewEpisodeMetricsInput,
    ViewOrdersInput,
    ViewPortfolioInput,
    ViewTimeInput,
    ViewUniverseInput,
)
from tradebench.episodes.tiers import TierId, describe_tier, load_tier

logger = logging.getLogger(__name__)


_ACTION_TO_INPUT_NONE: dict[str, type] = {
    "view_universe": ViewUniverseInput,
    "view_time": ViewTimeInput,
    "view_portfolio": ViewPortfolioInput,
    "view_orders": ViewOrdersInput,
    "view_constraints": ViewConstraintsInput,
    "view_episode_metrics": ViewEpisodeMetricsInput,
    "advance_day": AdvanceDayInput,
}

_ACTION_TO_INPUT_PAYLOAD: dict[str, type] = {
    "record_decision": RecordDecisionInput,
    "place_order": PlaceOrderInput,
    "cancel_order": CancelOrderInput,
    "sandbox_exec": SandboxExecInput,
}


_ALL_ACTION_TYPES = list(_ACTION_TO_INPUT_NONE) + list(_ACTION_TO_INPUT_PAYLOAD)


class TradeBenchEnvironment(Environment):
    """OpenEnv-compliant TradeBench environment.

    Each session holds one :class:`TradeBenchSessionRuntime`, keyed off a
    task tier selected at ``reset`` time. Concurrent sessions are
    supported because the runtime carries per-instance state (ledger,
    queued orders, current date); there is no module-level mutable state.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        super().__init__()
        self._state = TradeState(
            episode_id=str(uuid4()),
            step_count=0,
            task_tier="t1",
            current_date="",
            initial_cash=100_000.0,
            bars_total=0,
            bars_remaining=0,
            cumulative_reward=0.0,
            cumulative_log_wealth=0.0,
            violations_count=0,
            terminated_on_violation=False,
        )
        self._runtime: TradeBenchSessionRuntime | None = None

    def get_metadata(self) -> EnvironmentMetadata:
        readme_content: str | None = None
        for path in ("/app/README.md", "README.md"):
            try:
                with open(path, encoding="utf-8") as fh:
                    raw = fh.read()
                if raw.startswith("---"):
                    end = raw.find("---", 3)
                    if end != -1:
                        raw = raw[end + 3 :].lstrip("\n")
                readme_content = raw
                break
            except FileNotFoundError:
                continue

        return EnvironmentMetadata(
            name="TradeBench",
            description=(
                "Long-horizon finance trading RL environment (Theme 2). "
                "Dense log-wealth reward with multi-signal regularizers, "
                "layered look-ahead prevention, and reward-hack safeguards."
            ),
            version="0.1.0",
            readme_content=readme_content,
        )

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        task_tier: str = "t1",
        **kwargs: Any,
    ) -> TradeObservation:
        """Start a new episode on the given tier (``t1`` / ``train`` / ``test``)."""

        if self._runtime is not None:
            try:
                self._runtime.teardown()
            except Exception:  # noqa: BLE001 - best effort, new session takes over
                logger.exception("teardown of previous runtime failed")
            self._runtime = None

        tier = task_tier if task_tier in ("t1", "train", "test") else "t1"
        try:
            manifest = load_tier(tier)  # type: ignore[arg-type]
        except Exception as exc:  # noqa: BLE001 - surface to client
            return TradeObservation(
                done=True,
                reward=0.0,
                task_tier=tier,
                current_date="",
                bars_remaining=0,
                phase="error",
                tool_output=f"Failed to load tier {tier}: {exc}",
                available_actions=[],
                error=str(exc),
            )

        settings = resolve_settings()
        runtime = TradeBenchSessionRuntime(
            manifest=manifest,
            settings=settings,
            seed=seed,
        )
        runtime.setup()
        self._runtime = runtime

        bars_total = self._estimate_bars_total(runtime)
        initial_cash = float(manifest.initial_cash)

        self._state = TradeState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_tier=tier,  # type: ignore[arg-type]
            current_date=runtime.current_date.isoformat(),
            initial_cash=initial_cash,
            bars_total=bars_total,
            bars_remaining=bars_total,
            cumulative_reward=0.0,
            cumulative_log_wealth=0.0,
            violations_count=0,
            terminated_on_violation=False,
        )

        desc = describe_tier(tier)
        intro = (
            f"{desc['name']} — {desc['regime']} "
            f"({desc['horizon']}). {desc['intent']}\n"
            f"Episode window: {manifest.episode_start.isoformat()} → "
            f"{manifest.episode_end.isoformat()}. Initial cash: "
            f"${initial_cash:,.2f}. Universe size: "
            f"{len(manifest.universe_asset_ids)}."
        )

        carry_strategy = kwargs.get("carry_strategy") if kwargs else None
        if tier == "train":
            packet_path = (
                settings.dataset_root
                / "catalog"
                / manifest.dataset_version
                / "daily_bars"
                / "part-000.parquet"
            )
            try:
                packet = build_study_packet(manifest, bars_parquet_path=packet_path)
            except Exception as exc:  # noqa: BLE001 - surface to client
                logger.exception("study packet build failed")
                return TradeObservation(
                    done=True,
                    reward=0.0,
                    task_tier=tier,
                    current_date="",
                    bars_remaining=0,
                    phase="error",
                    tool_output=f"Study packet build failed: {exc}",
                    available_actions=[],
                    error=str(exc),
                )
            tool_output = (
                f"{intro}\n\n"
                "This is the TRAIN study tier. Read the study packet below, derive "
                "a strategy, then call `record_decision` exactly ONCE summarising "
                "the strategy you commit to. The episode terminates after that "
                "single call; no per-bar rollout, no reward.\n\n"
                f"{packet.text}"
            )
            return TradeObservation(
                done=False,
                reward=0.0,
                task_tier=tier,
                current_date=runtime.current_date.isoformat(),
                bars_remaining=0,
                phase="study",
                tool_output=tool_output,
                tool_metadata={
                    "manifest_task_id": manifest.task_id,
                    "episode_start": manifest.episode_start.isoformat(),
                    "episode_end": manifest.episode_end.isoformat(),
                    "universe": list(manifest.universe_asset_ids),
                    "study_packet_stats": packet.stats,
                    "study_packet_correlations": packet.correlations,
                    "study_packet_bars": packet.bars_count,
                },
                reward_breakdown={key: 0.0 for key in _REWARD_KEYS},
                portfolio_value=initial_cash,
                cash=initial_cash,
                positions={},
                available_actions=["record_decision"],
                violations=[],
                error=None,
            )

        if tier == "test" and isinstance(carry_strategy, str) and carry_strategy.strip():
            tool_output = (
                f"{intro}\n\n"
                "Strategy committed during the train study (verbatim from your "
                "record_decision):\n"
                f"{carry_strategy.strip()}"
            )
        else:
            tool_output = intro

        return TradeObservation(
            done=False,
            reward=0.0,
            task_tier=tier,
            current_date=runtime.current_date.isoformat(),
            bars_remaining=bars_total,
            phase="trading",
            tool_output=tool_output,
            tool_metadata={
                "manifest_task_id": manifest.task_id,
                "episode_start": manifest.episode_start.isoformat(),
                "episode_end": manifest.episode_end.isoformat(),
                "universe": list(manifest.universe_asset_ids),
            },
            reward_breakdown={key: 0.0 for key in _REWARD_KEYS},
            portfolio_value=initial_cash,
            cash=initial_cash,
            positions={},
            available_actions=list(_ALL_ACTION_TYPES),
            violations=[],
            error=None,
        )

    def step(
        self,
        action: TradeAction,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> TradeObservation:
        if self._runtime is None:
            return TradeObservation(
                done=True,
                reward=0.0,
                phase="error",
                tool_output="Environment not reset — call /reset first.",
                available_actions=[],
                error="runtime_not_initialized",
            )

        self._state.step_count += 1
        action_type = action.action_type

        if self._state.task_tier == "train" and action_type != "record_decision":
            return TradeObservation(
                done=False,
                reward=0.0,
                task_tier=self._state.task_tier,
                current_date=self._current_date_iso(),
                bars_remaining=0,
                phase="study",
                tool_output=(
                    "Train tier accepts only `record_decision`. Read the study "
                    "packet returned at reset, derive a strategy, then commit "
                    "via record_decision (which ends the train episode)."
                ),
                available_actions=["record_decision"],
                error=f"train_tier_only_record_decision:{action_type}",
            )

        try:
            tool_output = self._dispatch(action_type, action.payload or {})
        except ValueError as exc:
            return TradeObservation(
                done=False,
                reward=0.0,
                task_tier=self._state.task_tier,
                current_date=self._current_date_iso(),
                bars_remaining=self._state.bars_remaining,
                phase="error",
                tool_output=str(exc),
                available_actions=list(_ALL_ACTION_TYPES),
                error=f"invalid_payload:{action_type}",
            )
        except Exception as exc:  # noqa: BLE001 - surface runtime errors, don't crash server
            logger.exception("runtime dispatch failed for %s", action_type)
            return TradeObservation(
                done=False,
                reward=0.0,
                task_tier=self._state.task_tier,
                current_date=self._current_date_iso(),
                bars_remaining=self._state.bars_remaining,
                phase="error",
                tool_output=f"{type(exc).__name__}: {exc}",
                available_actions=list(_ALL_ACTION_TYPES),
                error="runtime_dispatch_failed",
            )

        text = _extract_text(tool_output)
        metadata = _extract_metadata(tool_output)
        reward = float(tool_output.reward) if tool_output.reward is not None else 0.0
        finished = bool(tool_output.finished)

        # Non-advance actions don't emit a full breakdown; fall back to a
        # wealth-only payload so the schema stays stable.
        breakdown_raw = metadata.get("reward_breakdown")
        if isinstance(breakdown_raw, dict) and breakdown_raw:
            reward_breakdown = {
                key: float(breakdown_raw.get(key, 0.0)) for key in _REWARD_KEYS
            }
        else:
            reward_breakdown = {key: 0.0 for key in _REWARD_KEYS}
            reward_breakdown["r_wealth"] = reward

        violations_raw = metadata.get("violations")
        violations: list[str] = (
            [str(v) for v in violations_raw]
            if isinstance(violations_raw, list)
            else []
        )
        if violations:
            self._state.violations_count += len(violations)
            if reward_breakdown.get("r_hack", 0.0) <= -1.0 or reward_breakdown.get(
                "r_rules",
                0.0,
            ) <= -1.0:
                self._state.terminated_on_violation = True

        self._state.cumulative_reward += reward
        self._state.cumulative_log_wealth += reward_breakdown.get("r_wealth", 0.0)
        if action_type == "advance_day":
            self._state.bars_remaining = max(0, self._state.bars_remaining - 1)
        self._state.current_date = self._current_date_iso()

        # On the train tier, a successful record_decision ends the study
        # episode immediately — no per-bar rollout, no advance_day, no reward.
        if (
            self._state.task_tier == "train"
            and action_type == "record_decision"
            and not metadata.get("error")
        ):
            finished = True
            self._state.bars_remaining = 0

        if finished:
            self._state.bars_remaining = 0
            # Keep the runtime around for a final /state read; teardown on next reset.

        portfolio = _snapshot_portfolio(self._runtime, metadata)

        return TradeObservation(
            done=finished,
            reward=reward,
            task_tier=self._state.task_tier,
            current_date=self._state.current_date,
            bars_remaining=self._state.bars_remaining,
            phase="done" if finished else "trading",
            tool_output=text,
            tool_metadata=metadata,
            reward_breakdown=reward_breakdown,
            portfolio_value=portfolio["value"],
            cash=portfolio["cash"],
            positions=portfolio["positions"],
            available_actions=list(_ALL_ACTION_TYPES),
            violations=violations,
            error=None,
        )

    @property
    def state(self) -> TradeState:
        return self._state

    def _dispatch(self, action_type: str, payload: dict[str, Any]) -> Any:
        if self._runtime is None:
            msg = "runtime not initialized"
            raise RuntimeError(msg)

        if action_type in _ACTION_TO_INPUT_NONE:
            input_cls = _ACTION_TO_INPUT_NONE[action_type]
            tool_input = input_cls()
            tool_fn = getattr(self._runtime, action_type)
            return tool_fn(tool_input)

        if action_type in _ACTION_TO_INPUT_PAYLOAD:
            input_cls = _ACTION_TO_INPUT_PAYLOAD[action_type]
            try:
                tool_input = input_cls(**payload)
            except Exception as exc:  # pydantic ValidationError et al.
                msg = (
                    f"invalid payload for action_type={action_type}: {exc}. "
                    f"expected schema {input_cls.__name__}"
                )
                raise ValueError(msg) from exc
            tool_fn = getattr(self._runtime, action_type)
            return tool_fn(tool_input)

        msg = (
            f"Unknown action_type: {action_type!r}. "
            f"Expected one of: {_ALL_ACTION_TYPES}"
        )
        raise ValueError(msg)

    def _current_date_iso(self) -> str:
        if self._runtime is None:
            return ""
        return self._runtime.current_date.isoformat()

    def _estimate_bars_total(self, runtime: TradeBenchSessionRuntime) -> int:
        """Approximate episode length in bars via a linear weekday count.

        Avoids the full ``pit.next_session_after`` walk at reset time; close
        enough for UI / state surfacing.
        """

        start = runtime.manifest.episode_start
        end = runtime.manifest.episode_end
        delta_days = (end - start).days
        if delta_days <= 0:
            return 0
        weekdays = sum(
            1
            for i in range(delta_days + 1)
            if (start.toordinal() + i) % 7 < 5
        )
        return weekdays


_REWARD_KEYS = (
    "r_wealth",
    "r_sharpe_bonus",
    "r_drawdown",
    "r_turnover",
    "r_concentration",
    "r_rules",
    "r_hack",
)


def _extract_text(tool_output: Any) -> str:
    blocks = getattr(tool_output, "blocks", None) or []
    pieces: list[str] = []
    for block in blocks:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            pieces.append(text)
    return "\n".join(pieces)


def _extract_metadata(tool_output: Any) -> dict[str, Any]:
    metadata = getattr(tool_output, "metadata", None)
    if isinstance(metadata, dict):
        return dict(metadata)
    return {}


def _snapshot_portfolio(
    runtime: TradeBenchSessionRuntime,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Flatten the portfolio into ``{value, cash, positions}`` for observation.

    Always computes mark-to-market ``portfolio_value`` via the runtime's
    metrics snapshot so the value stays consistent across tools (some tools,
    e.g. ``record_decision``, don't carry a ``metrics`` metadata block).
    """

    portfolio_meta = metadata.get("portfolio") if isinstance(metadata, dict) else None
    cash: float
    positions: dict[str, float]
    if isinstance(portfolio_meta, dict):
        cash = _coerce_float(portfolio_meta.get("cash"), default=0.0)
        positions = _coerce_positions(
            portfolio_meta.get("positions_by_asset")
            or portfolio_meta.get("positions"),
        )
    else:
        try:
            state = runtime._portfolio_state()  # type: ignore[attr-defined]  # noqa: SLF001
            cash = _coerce_float(getattr(state, "cash", 0.0), default=0.0)
            positions = _coerce_positions(getattr(state, "positions_by_asset", {}))
        except Exception:  # noqa: BLE001
            cash = 0.0
            positions = {}

    # Prefer metadata-supplied value when advance_day computed it; otherwise
    # fall back to the runtime's metrics snapshot.
    metrics_meta = metadata.get("metrics") if isinstance(metadata, dict) else None
    value_val: Any = None
    if isinstance(metrics_meta, dict):
        value_val = metrics_meta.get("portfolio_value")
    if value_val is None:
        try:
            snapshot = runtime._metrics_snapshot()  # type: ignore[attr-defined]  # noqa: SLF001
            value_val = getattr(snapshot, "portfolio_value", None)
        except Exception:  # noqa: BLE001
            value_val = None

    return {
        "cash": cash,
        "positions": positions,
        "value": _coerce_float(value_val, default=cash),
    }


def _coerce_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_positions(raw: Any) -> dict[str, float]:
    """Coerce a positions-by-asset map to ``dict[str, float]`` (share count).

    Accepts ``Position`` objects, Pydantic-serialized dicts (``shares`` or
    ``quantity``), or raw scalars.
    """

    if not isinstance(raw, dict):
        return {}
    out: dict[str, float] = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            scalar = value.get("shares")
            if scalar is None:
                scalar = value.get("quantity")
        else:
            scalar = getattr(value, "shares", None)
            if scalar is None:
                scalar = getattr(value, "quantity", value)
        out[str(key)] = _coerce_float(scalar, default=0.0)
    return out


__all__ = ["TradeBenchEnvironment"]
