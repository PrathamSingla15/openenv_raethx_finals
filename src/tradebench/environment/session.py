"""Session runtime: the authoritative tool dispatcher for one TradeBench episode.

Consumed by ``server/tradebench_environment.py`` (the OpenEnv ``Environment``
subclass). Every tool method takes a frozen Pydantic input from ``.tools``
and returns a :class:`ToolOutput` defined in :mod:`._io`.
"""

from __future__ import annotations

import math
import random
from collections import deque
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel

from tradebench.data.catalog import DatasetCatalog
from tradebench.data.query import PitQueryService
from tradebench.episodes.models import (
    ConvictionEntry,
    DecisionSnapshot,
    EpisodeManifest,
    manifest_digest,
)
from tradebench.execution.costs import CostModel
from tradebench.execution.engine import (
    OrderIntent,
    advance_trading_session,
    portfolio_market_value,
)
from tradebench.ledger.events import (
    DecisionRecorded,
    DividendApplied,
    LedgerEvent,
    OrderCancelled,
    OrderFilled,
    OrderRejected,
)
from tradebench.ledger.projector import project
from tradebench.ledger.state import PortfolioState
from tradebench.rewards.anti_hack import (
    check_rules_clause,
    scan_forbidden_globals,
)
from tradebench.rewards.composite import RewardBreakdown, compute_composite_reward
from tradebench.sandbox.models import (
    EpisodeWorkspace,
    SandboxRunner,
    SandboxRunRequest,
    SandboxRunResult,
)
from tradebench.sandbox.providers.docker import DockerSandboxRunner
from tradebench.sandbox.providers.local import LocalSandboxRunner
from tradebench.sandbox.workspace import materialize_episode_workspace
from tradebench.settings import TradeBenchSettings

from ._io import JSONObject, TextBlock, ToolOutput
from .date_gate import LookaheadViolation, check_date_bound
from .progressive_fs import materialize_allowed_files, refresh_allowed_files
from .metrics import (
    EpisodeMetricsSnapshot,
    build_episode_metrics_snapshot,
    compute_realized_hhi,
    compute_traded_notional,
    portfolio_marks_at_close,
)
from .prompt import build_prompt
from .tools import (
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

JsonValue = object

DEFAULT_COST_MODEL = CostModel(
    commission_floor=Decimal("1"),
    fee_bps=Decimal("10"),
    slippage_bps_cap=Decimal("50"),
)

BOOTSTRAP_ASSET_ID = "__tradebench_initial_cash__"


class ToolCallFailure(RuntimeError):
    """Raised when the tool dispatcher rejects a call."""


def _tool_output(
    text: str,
    *,
    metadata: dict[str, JsonValue] | None = None,
    detail: dict[str, JsonValue] | None = None,
    reward: float | None = None,
    finished: bool = False,
) -> ToolOutput:
    return ToolOutput(
        blocks=[
            TextBlock(
                text=text,
                detail=None if detail is None else cast(JSONObject, detail),
            ),
        ],
        metadata=None if metadata is None else cast(JSONObject, metadata),
        reward=reward,
        finished=finished,
    )


def _model_json(model: BaseModel) -> dict[str, JsonValue]:
    return cast(dict[str, JsonValue], model.model_dump(mode="json"))


def _decimal_string_map(values: dict[str, Decimal]) -> dict[str, object]:
    return {key: str(value) for key, value in values.items()}


class TradeBenchSessionRuntime:
    """Authoritative state for one local TradeBench environment session."""

    def __init__(
        self,
        manifest: EpisodeManifest,
        settings: TradeBenchSettings,
        *,
        seed: int | None = None,
    ) -> None:
        self.manifest = manifest
        self.settings = settings
        self.episode_id = manifest_digest(manifest)
        self.current_date = manifest.episode_start

        # When ``seed`` is None we derive a stable default from the manifest
        # digest so two runs of the same episode behave identically.
        if seed is None:
            seed = int(self.episode_id[:16], 16)
        self._rng: random.Random = random.Random(seed)
        self._seed: int = seed

        self._catalog = DatasetCatalog(settings.dataset_root, manifest.dataset_version)
        self._pit: PitQueryService | None = None
        self._workspace: EpisodeWorkspace | None = None
        self._sandbox_runner: SandboxRunner | None = None

        self._events: list[LedgerEvent] = []
        self._queued_orders: list[OrderIntent] = []
        self._decisions_by_date: dict[date, DecisionSnapshot] = {}
        self._manual_event_counter = 0

        # ``_recent_log_returns`` holds the last 20 bar-level agent log
        # returns. ``_recent_bar_alphas`` mirrors it for ``r_t^p - r_t^b``,
        # consumed by ``c_consistency``'s downside-semideviation. The
        # benchmark trajectory is anchored once at ``setup()`` and stepped
        # forward each ``advance_day``. Pending-violation lists are populated
        # by sandbox_exec / record_decision and drained on the next advance_day.
        self._recent_log_returns: deque[float] = deque(maxlen=20)
        self._recent_bar_alphas: deque[float] = deque(maxlen=20)
        self._high_watermark: Decimal = manifest.initial_cash
        self._violations_hack_pending: list[str] = []
        self._violations_rules_pending: list[str] = []
        self._reward_inputs_log: list[dict[str, Any]] = []
        self._reward_breakdown_log: list[dict[str, float]] = []
        self._per_bar_rewards: list[float] = []
        self._benchmark_quantities: dict[str, Decimal] | None = None
        self._benchmark_initial_value: Decimal = manifest.initial_cash
        self._last_benchmark_value: Decimal = manifest.initial_cash

    def setup(self) -> None:
        if self._pit is not None:
            return

        self._catalog.validate_layout()
        self._pit = PitQueryService(self._catalog)
        self._workspace = self._materialize_workspace()
        self._events = [self._bootstrap_cash_event()]
        self._anchor_benchmark()

    def teardown(self) -> None:
        if self._pit is not None:
            self._pit.close()
        self._pit = None
        self._workspace = None
        self._sandbox_runner = None

    @property
    def workspace(self) -> EpisodeWorkspace:
        if self._workspace is None:
            msg = "session runtime is not set up"
            raise RuntimeError(msg)
        return self._workspace

    @property
    def pit(self) -> PitQueryService:
        if self._pit is None:
            msg = "session runtime is not set up"
            raise RuntimeError(msg)
        return self._pit

    @property
    def sandbox_runner(self) -> SandboxRunner:
        if self._workspace is None:
            msg = "session runtime is not set up"
            raise RuntimeError(msg)
        if self._sandbox_runner is None:
            self._sandbox_runner = self._make_sandbox_runner()
        return self._sandbox_runner

    def ledger_events(self) -> tuple[LedgerEvent, ...]:
        """Return an immutable snapshot of the authoritative episode event log."""

        return tuple(self._events)

    def run_in_sandbox(
        self,
        command: list[str],
        *,
        env: dict[str, str] | None = None,
        timeout_seconds: int | None = None,
    ) -> SandboxRunResult:
        """Run one command inside the episode sandbox using the mounted workspace."""

        return self.sandbox_runner.run(
            SandboxRunRequest(
                command=command,
                episode_workspace=self.workspace,
                env={} if env is None else dict(env),
                timeout_seconds=timeout_seconds,
            ),
        )

    def prompt_blocks(self) -> list[TextBlock]:
        visible_universe = self._visible_universe_rows()
        return build_prompt(
            manifest=self.manifest,
            current_date=self.current_date,
            next_session_date=self._bounded_next_session(),
            workspace=self.workspace,
            visible_universe_count=len(visible_universe),
        )

    def view_universe(self, _input: ViewUniverseInput) -> ToolOutput:
        if (gate_err := self._gate_as_of(_input.as_of_date)) is not None:
            return gate_err
        as_of = _input.as_of_date or self.current_date
        universe = self._visible_universe_rows(as_of=as_of)
        header = (
            f"Visible universe as of {as_of.isoformat()}:"
            if _input.as_of_date is not None
            else "Visible universe for the current decision date:"
        )
        lines = [header]
        lines.extend(
            f"- {row['symbol']} ({row['asset_id']})"
            if row["symbol"] is not None
            else f"- {row['asset_id']}"
            for row in universe
        )
        return _tool_output(
            "\n".join(lines),
            metadata={
                "session_date": self.current_date.isoformat(),
                "as_of_date": as_of.isoformat(),
                "universe": universe,
            },
        )

    def view_time(self, _input: ViewTimeInput) -> ToolOutput:
        next_session = self._bounded_next_session()
        payload = {
            "current_decision_date": self.current_date.isoformat(),
            "next_execution_date": (
                None if next_session is None else next_session.isoformat()
            ),
            "episode_start": self.manifest.episode_start.isoformat(),
            "episode_end": self.manifest.episode_end.isoformat(),
        }
        return _tool_output(
            (
                f"Current decision date: {payload['current_decision_date']}\n"
                f"Next execution date: {payload['next_execution_date']}"
            ),
            metadata={"time": payload},
        )

    def view_portfolio(self, _input: ViewPortfolioInput) -> ToolOutput:
        state = self._portfolio_state()
        metrics = self._metrics_snapshot()
        marks = portfolio_marks_at_close(
            pit=self.pit,
            portfolio=state,
            session_date=self.current_date,
        )
        return _tool_output(
            (
                f"Cash: {state.cash}\n"
                f"Positions: {len(state.positions_by_asset)}\n"
                f"Portfolio value: {metrics.portfolio_value}"
            ),
            metadata={
                "portfolio": _model_json(state),
                "marks": _decimal_string_map(marks),
                "metrics": _model_json(metrics),
            },
        )

    def view_orders(self, _input: ViewOrdersInput) -> ToolOutput:
        state = self._portfolio_state()
        queued = [_model_json(order) for order in self._queued_orders]
        open_orders = [_model_json(order) for order in state.open_orders.values()]
        return _tool_output(
            (
                f"Queued orders: {len(queued)}\n"
                f"Ledger open orders: {len(open_orders)}"
            ),
            metadata={
                "orders": {
                    "queued": queued,
                    "open": open_orders,
                },
            },
        )

    def view_constraints(self, _input: ViewConstraintsInput) -> ToolOutput:
        policy = (
            {}
            if self.manifest.constraint_policy is None
            else self.manifest.constraint_policy.model_dump(mode="json")
        )
        constraints: dict[str, JsonValue] = {
            "long_only": True,
            "shorting_allowed": False,
            "leverage_allowed": False,
            "integer_shares_only": True,
            "decision_required_before_first_order": True,
            "decision_required_per_bar": True,
            "constraint_policy": cast(dict[str, JsonValue], policy),
        }
        return _tool_output(
            "TradeBench constraints are active for this episode.",
            metadata={"constraints": constraints},
        )

    def record_decision(self, params: RecordDecisionInput) -> ToolOutput:
        if self.current_date in self._decisions_by_date:
            return self._structured_error(
                code="decision_already_recorded",
                message="A decision has already been recorded for this session date.",
                extra={"session_date": self.current_date.isoformat()},
            )

        unknown_assets = sorted(
            conviction.asset_id
            for conviction in params.top_convictions
            if conviction.asset_id not in set(self.manifest.universe_asset_ids)
        )
        if unknown_assets:
            return self._structured_error(
                code="unknown_conviction_asset",
                message=(
                    "Decision snapshot referenced assets outside the manifest "
                    "universe."
                ),
                extra={"asset_ids": unknown_assets},
            )

        snapshot = DecisionSnapshot(
            regime_label=params.regime_label,
            edge_summary=params.edge_summary,
            intended_exposure=params.intended_exposure,
            top_convictions=[
                ConvictionEntry(asset_id=item.asset_id, weight=item.weight)
                for item in params.top_convictions
            ],
            uncertainty=params.uncertainty,
        )
        self._decisions_by_date[self.current_date] = snapshot
        event_id, event_time = self._next_manual_event("decision")
        self._events.append(
            DecisionRecorded(
                event_id=event_id,
                episode_id=self.episode_id,
                event_time=event_time,
                snapshot=snapshot,
            ),
        )

        # Rules-clause scan over agent free-text. Hits accumulate and are
        # drained at the next advance_day (matching the cadence of r_hack).
        scan_text = "\n".join(
            part
            for part in (params.reasoning, params.edge_summary)
            if part
        )
        rules_hits = check_rules_clause(scan_text) if scan_text else []
        if rules_hits:
            self._violations_rules_pending.extend(rules_hits)

        message = f"Recorded decision for {self.current_date.isoformat()}."
        if rules_hits:
            message += (
                "\nVIOLATION: rules-clause pattern detected — r_rules=-1.0 will "
                "apply on the next advance_day and the episode will terminate."
            )

        return _tool_output(
            message,
            metadata={
                "recorded": True,
                "session_date": self.current_date.isoformat(),
                "decision": _model_json(snapshot),
                "violations": list(rules_hits),
            },
        )

    def place_order(self, params: PlaceOrderInput) -> ToolOutput:
        if self.current_date not in self._decisions_by_date:
            return self._structured_error(
                code="decision_required",
                message=(
                    "Call record_decision before the first order of this "
                    "session date."
                ),
                extra={"session_date": self.current_date.isoformat()},
                accepted=False,
            )

        if self._bounded_next_session() is None:
            return self._structured_error(
                code="episode_complete",
                message="No further execution session is available for this episode.",
                extra={"session_date": self.current_date.isoformat()},
                accepted=False,
                finished=True,
            )

        if params.client_order_id in self._known_client_order_ids():
            return self._structured_error(
                code="duplicate_client_order_id",
                message="client_order_id must be unique within an episode.",
                extra={"client_order_id": params.client_order_id},
                accepted=False,
            )

        order = OrderIntent(
            client_order_id=params.client_order_id,
            asset_id=params.asset_id,
            side=params.side,
            quantity=params.quantity,
        )
        self._queued_orders.append(order)
        return _tool_output(
            (
                f"Queued order {params.client_order_id} "
                f"for execution after {self.current_date.isoformat()}."
            ),
            metadata={
                "accepted": True,
                "session_date": self.current_date.isoformat(),
                "queued_order": _model_json(order),
            },
        )

    def cancel_order(self, params: CancelOrderInput) -> ToolOutput:
        for index, order in enumerate(self._queued_orders):
            if order.client_order_id == params.client_order_id:
                removed = self._queued_orders.pop(index)
                return _tool_output(
                    f"Cancelled queued order {params.client_order_id}.",
                    metadata={
                        "cancelled": True,
                        "location": "queued",
                        "order": _model_json(removed),
                    },
                )

        state = self._portfolio_state()
        open_order = state.open_orders.get(params.client_order_id)
        if open_order is not None:
            event_id, event_time = self._next_manual_event("cancel")
            self._events.append(
                OrderCancelled(
                    event_id=event_id,
                    episode_id=self.episode_id,
                    event_time=event_time,
                    client_order_id=params.client_order_id,
                ),
            )
            return _tool_output(
                f"Cancelled open order {params.client_order_id}.",
                metadata={
                    "cancelled": True,
                    "location": "ledger",
                    "order": _model_json(open_order),
                },
            )

        return self._structured_error(
            code="order_not_found",
            message="No queued or open order matched client_order_id.",
            extra={"client_order_id": params.client_order_id},
            accepted=False,
        )

    def advance_day(self, _input: AdvanceDayInput) -> ToolOutput:
        next_session = self._bounded_next_session()
        if next_session is None:
            return self._structured_error(
                code="episode_complete",
                message="No additional session is available to advance into.",
                extra={"session_date": self.current_date.isoformat()},
                finished=True,
            )

        if self.current_date not in self._decisions_by_date:
            return self._structured_error(
                code="decision_required",
                message=(
                    "Long-horizon planning gate: every bar requires a fresh "
                    "record_decision before advance_day. Call record_decision "
                    "with regime_label, edge_summary, intended_exposure, "
                    "top_convictions, and uncertainty for the current "
                    f"session_date={self.current_date.isoformat()}, then "
                    "advance_day."
                ),
                extra={"session_date": self.current_date.isoformat()},
            )

        decision_date = self.current_date
        # Capture V_before at the decision-date close before
        # advance_trading_session runs — the composite reward needs both
        # endpoints of the bar.
        value_before = self._value_at_close(decision_date)
        prior_high_watermark = self._high_watermark
        try:
            step_events = advance_trading_session(
                pit=self.pit,
                manifest=self.manifest,
                episode_id=self.episode_id,
                decision_date=decision_date,
                execution_date=next_session,
                prior_events=self._events,
                new_orders=list(self._queued_orders),
                cost_model=DEFAULT_COST_MODEL,
                base_event_time=datetime.combine(
                    decision_date,
                    time(hour=16, tzinfo=UTC),
                ),
            )
        except (TypeError, ValueError) as exc:
            return self._structured_error(
                code="advance_failed",
                message=str(exc),
                extra={
                    "decision_date": decision_date.isoformat(),
                    "execution_date": next_session.isoformat(),
                },
            )

        self._events.extend(step_events)
        self._queued_orders.clear()
        self.current_date = next_session

        if self._workspace is not None:
            refresh_allowed_files(
                source_root=self._catalog.version_dir,
                dest_root=self._workspace.host_data_dir,
                new_current_date=self.current_date,
            )

        metrics = self._metrics_snapshot()
        fills = [
            _model_json(event)
            for event in step_events
            if isinstance(event, OrderFilled)
        ]
        rejections = [
            _model_json(event)
            for event in step_events
            if isinstance(event, OrderRejected)
        ]

        # V_after is the portfolio mark at the next-session close (post-fill,
        # post-corp-actions). HHI is computed over the realized risky-sleeve
        # weights at that close.
        value_after = Decimal(metrics.portfolio_value)
        traded_notional = compute_traded_notional(step_events)
        turnover_ratio = (
            float(traded_notional / value_before) if value_before > 0 else 0.0
        )
        state_after = self._portfolio_state()
        marks_after = (
            portfolio_marks_at_close(
                pit=self.pit,
                portfolio=state_after,
                session_date=next_session,
            )
            if state_after.positions_by_asset
            else {}
        )
        hhi = compute_realized_hhi(
            state=state_after,
            marks=marks_after,
            portfolio_value=value_after,
        )
        violations_drained: list[str] = (
            list(self._violations_rules_pending) + list(self._violations_hack_pending)
        )

        v0 = self._benchmark_initial_value
        bench_before = self._last_benchmark_value
        bench_after = self._benchmark_value_at(self.current_date)
        cum_log_return = (
            math.log(float(value_after) / float(v0)) if value_after > 0 else -10.0
        )
        cum_bench_log = (
            math.log(float(bench_after) / float(v0)) if bench_after > 0 else 0.0
        )
        bar_log_return = (
            math.log(float(value_after) / float(value_before))
            if value_before > 0 and value_after > 0
            else 0.0
        )
        bar_bench_log = (
            math.log(float(bench_after) / float(bench_before))
            if bench_before > 0 and bench_after > 0
            else 0.0
        )
        bar_alpha = bar_log_return - bar_bench_log
        gross_leverage = self._gross_leverage(value_after)

        reward_inputs = {
            "value_after": value_after,
            "initial_value": v0,
            "cumulative_log_return": cum_log_return,
            "cumulative_benchmark_log_return": cum_bench_log,
            "recent_bar_alphas": tuple(self._recent_bar_alphas),
            "high_watermark": float(prior_high_watermark),
            "turnover_ratio": turnover_ratio,
            "hhi": hhi,
            "gross_leverage": gross_leverage,
            "violations_rules": bool(self._violations_rules_pending),
            "violations_hack": bool(self._violations_hack_pending),
        }
        self._reward_inputs_log.append(reward_inputs)
        try:
            breakdown = compute_composite_reward(**reward_inputs)
        except ValueError:
            breakdown = RewardBreakdown(
                c_alpha=0.0,
                c_return=0.0,
                c_drawdown=0.0,
                c_solvency=0.0,
                c_efficiency=1.0,
                c_diversity=1.0,
                c_consistency=0.5,
                g_compliance=0.0,
                r_total=0.0,
            )

        self._violations_rules_pending.clear()
        self._violations_hack_pending.clear()

        self._recent_log_returns.append(bar_log_return)
        self._recent_bar_alphas.append(bar_alpha)
        if value_after > self._high_watermark:
            self._high_watermark = value_after

        reward = breakdown.total()
        self._per_bar_rewards.append(reward)
        reward_breakdown_payload = breakdown.to_dict()
        self._reward_breakdown_log.append(dict(reward_breakdown_payload))
        score_normalized = (
            sum(self._per_bar_rewards) / len(self._per_bar_rewards)
            if self._per_bar_rewards
            else 0.5
        )
        finished = self._bounded_next_session() is None
        return _tool_output(
            (
                "Advanced from "
                f"{decision_date.isoformat()} to {next_session.isoformat()}.\n"
                f"Fills: {len(fills)}\n"
                f"Rejections: {len(rejections)}"
            ),
            metadata={
                "decision_date": decision_date.isoformat(),
                "execution_date": next_session.isoformat(),
                "fills": fills,
                "rejections": rejections,
                "portfolio": _model_json(self._portfolio_state()),
                "metrics": _model_json(metrics),
                "reward_breakdown": reward_breakdown_payload,
                "violations": violations_drained,
                "score_normalized": score_normalized,
                "benchmark_value": str(bench_after),
                "cumulative_log_alpha": cum_log_return - cum_bench_log,
            },
            reward=reward,
            finished=finished,
        )

    def view_episode_metrics(self, _input: ViewEpisodeMetricsInput) -> ToolOutput:
        # Snapshots project up to ``current_date``; past as_of queries are not
        # supported, but the gate still rejects future requests.
        if (gate_err := self._gate_as_of(_input.as_of_date)) is not None:
            return gate_err
        metrics = self._metrics_snapshot()
        return _tool_output(
            f"Current episode score: {metrics.score:.12f}",
            metadata={"metrics": _model_json(metrics)},
            finished=self._bounded_next_session() is None,
        )

    def sandbox_exec(self, params: SandboxExecInput) -> ToolOutput:
        """Execute a command inside the episode sandbox and return output."""

        result = self.run_in_sandbox(
            params.command,
            env=params.env if params.env else None,
            timeout_seconds=params.timeout_seconds,
        )
        text_lines = [
            f"exit_code: {result.exit_code}",
        ]
        if result.stdout:
            text_lines.append(f"stdout:\n{result.stdout}")
        if result.stderr:
            text_lines.append(f"stderr:\n{result.stderr}")

        # Forbidden-global scan over stdout + stderr. Hits persist across
        # tool calls so the next advance_day consumes them into r_hack.
        combined_text = "\n".join(
            part for part in (result.stdout, result.stderr) if part
        )
        hack_hits = scan_forbidden_globals(combined_text)
        if hack_hits:
            self._violations_hack_pending.extend(hack_hits)
            text_lines.append(
                "VIOLATION: forbidden-global pattern detected — r_hack=-1.0 will "
                "apply on the next advance_day and the episode will terminate.",
            )

        return _tool_output(
            "\n".join(text_lines),
            metadata={
                "exit_code": result.exit_code,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "violations": list(hack_hits),
            },
        )

    def _visible_universe_rows(
        self,
        *,
        as_of: date | None = None,
    ) -> list[dict[str, object]]:
        target = as_of if as_of is not None else self.current_date
        asset_ids = self.pit.load_universe(target, self.manifest)
        symbol_to_asset = self.pit.get_symbol_to_asset_id(target, asset_ids)
        asset_to_symbol = {
            asset_id: symbol for symbol, asset_id in symbol_to_asset.items()
        }
        return [
            {
                "asset_id": asset_id,
                "symbol": asset_to_symbol.get(asset_id),
            }
            for asset_id in asset_ids
        ]

    def _gate_as_of(self, requested: date | None) -> ToolOutput | None:
        """Reject as_of requests beyond ``self.current_date``.

        Returns ``None`` when the request is omitted or in-bounds — the
        caller proceeds normally. Returns a structured-error ``ToolOutput``
        when the request would peek into the future, which the caller
        should return verbatim. Mirrors the behavior of
        :func:`tradebench.environment.date_gate.check_date_bound` but emits
        the env-standard structured error instead of raising.
        """

        if requested is None:
            return None
        try:
            check_date_bound(requested, self.current_date)
        except LookaheadViolation as exc:
            requested_iso = (
                requested.isoformat()
                if isinstance(requested, date)
                else str(requested)
            )
            return self._structured_error(
                code="lookahead_violation",
                message=str(exc),
                extra={
                    "requested": requested_iso,
                    "current_date": self.current_date.isoformat(),
                },
            )
        return None

    def _portfolio_state(self) -> PortfolioState:
        return project(self._events)

    def _value_at_close(self, session_date: date) -> Decimal:
        """Mark the current ledger to the supplied session-date close.

        Used to capture ``V_before`` (decision-date close) before
        ``advance_trading_session`` runs and ``V_after`` (next-session
        close) after it, both as :class:`Decimal` for the composite-reward
        inputs.
        """

        state = self._portfolio_state()
        if not state.positions_by_asset:
            return state.cash
        marks = portfolio_marks_at_close(
            pit=self.pit,
            portfolio=state,
            session_date=session_date,
        )
        return portfolio_market_value(state, marks)

    def _metrics_snapshot(self) -> EpisodeMetricsSnapshot:
        return build_episode_metrics_snapshot(
            pit=self.pit,
            manifest=self.manifest,
            current_date=self.current_date,
            next_session_date=self._bounded_next_session(),
            events=self._events,
            queued_orders_count=len(self._queued_orders),
        )

    def _anchor_benchmark(self) -> None:
        """Pin the equal-weight buy-and-hold benchmark at ``episode_start``.

        Allocates ``initial_cash / N`` to each universe asset at the
        episode-start close and records the per-asset share quantities. The
        benchmark trajectory is recomputed each ``advance_day`` by re-marking
        these quantities at the new close.
        """

        universe = self.pit.load_universe(self.manifest.episode_start, self.manifest)
        if not universe:
            self._benchmark_quantities = {}
            return
        frame = self.pit.get_bars(
            list(universe),
            self.manifest.episode_start,
            lookback_days=1,
        )
        per_asset_value = self.manifest.initial_cash / Decimal(len(universe))
        quantities: dict[str, Decimal] = {}
        for asset_id in universe:
            rows = frame[
                (frame["asset_id"] == asset_id)
                & (
                    frame["session_date"].map(
                        lambda d: d.date() if hasattr(d, "date") else d,
                    )
                    == self.manifest.episode_start
                )
            ]
            if rows.empty:
                continue
            close = Decimal(str(rows.iloc[-1]["close"]))
            if close <= 0:
                continue
            quantities[asset_id] = per_asset_value / close
        self._benchmark_quantities = quantities

    def _benchmark_value_at(self, session_date: date) -> Decimal:
        """Mark the equal-weight B&H portfolio at ``session_date`` close."""

        if not self._benchmark_quantities:
            return self.manifest.initial_cash
        asset_ids = list(self._benchmark_quantities)
        frame = self.pit.get_bars(asset_ids, session_date, lookback_days=1)
        total = Decimal("0")
        for asset_id in asset_ids:
            rows = frame[
                (frame["asset_id"] == asset_id)
                & (
                    frame["session_date"].map(
                        lambda d: d.date() if hasattr(d, "date") else d,
                    )
                    == session_date
                )
            ]
            if rows.empty:
                continue
            close = Decimal(str(rows.iloc[-1]["close"]))
            total += self._benchmark_quantities[asset_id] * close
        if total <= 0:
            return self._last_benchmark_value
        self._last_benchmark_value = total
        return total

    def _gross_leverage(self, value_after: Decimal) -> float:
        if value_after <= 0:
            return 0.0
        state = self._portfolio_state()
        if not state.positions_by_asset:
            return 0.0
        marks = portfolio_marks_at_close(
            pit=self.pit,
            portfolio=state,
            session_date=self.current_date,
        )
        gross = Decimal("0")
        for asset_id, position in state.positions_by_asset.items():
            mv = abs(Decimal(position.shares) * marks[asset_id])
            gross += mv
        return float(gross / value_after)

    def _bounded_next_session(self) -> date | None:
        next_session = self.pit.next_session_after(self.current_date)
        if next_session is None or next_session > self.manifest.episode_end:
            return None
        return next_session

    def _known_client_order_ids(self) -> set[str]:
        known = {order.client_order_id for order in self._queued_orders}
        for event in self._events:
            client_order_id = getattr(event, "client_order_id", None)
            if isinstance(client_order_id, str):
                known.add(client_order_id)
        return known

    def _bootstrap_cash_event(self) -> DividendApplied:
        return DividendApplied(
            event_id=f"{self.episode_id}:bootstrap_cash",
            episode_id=self.episode_id,
            event_time=datetime.combine(
                self.manifest.episode_start,
                time(hour=9, tzinfo=UTC),
            ),
            asset_id=BOOTSTRAP_ASSET_ID,
            cash_credited=self.manifest.initial_cash,
        )

    def _next_manual_event(self, suffix: str) -> tuple[str, datetime]:
        self._manual_event_counter += 1
        timestamp = datetime.combine(
            self.current_date,
            time(hour=16, tzinfo=UTC),
        ) + timedelta(microseconds=self._manual_event_counter)
        event_id = (
            f"{self.episode_id}:{self.current_date.isoformat()}:manual:"
            f"{self._manual_event_counter:05d}:{suffix}"
        )
        return event_id, timestamp

    def _materialize_workspace(self) -> EpisodeWorkspace:
        manifest_dir = self.settings.artifact_root / "_manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_dir / f"{self.episode_id}.json"
        manifest_path.write_text(
            self.manifest.model_dump_json(indent=2),
            encoding="utf-8",
        )

        # Progressive filesystem: populate a per-episode ``data/`` directory
        # with only those files whose inferred partition_date is at or before
        # the current simulation date. ``advance_day`` then refreshes it.
        filtered_data_dir = (
            self.settings.artifact_root
            / "sandboxes"
            / self.episode_id
            / "data"
        )
        filtered_data_dir.mkdir(parents=True, exist_ok=True)
        materialize_allowed_files(
            source_root=self._catalog.version_dir,
            dest_root=filtered_data_dir,
            current_date=self.current_date,
        )

        return materialize_episode_workspace(
            artifact_root=self.settings.artifact_root,
            episode_id=self.episode_id,
            dataset_host_path=filtered_data_dir,
            manifest_source_path=manifest_path,
        )

    def _make_sandbox_runner(self) -> SandboxRunner:
        if self.settings.sandbox_provider == "local":
            return LocalSandboxRunner()

        import docker  # type: ignore[import-not-found,import-untyped]

        seccomp = (
            self._project_root()
            / "docker"
            / "constraints"
            / "seccomp.json"
        )
        return DockerSandboxRunner(
            docker.from_env(),
            image=self.settings.sandbox_image,
            seccomp_profile_host_path=seccomp,
        )

    def _project_root(self) -> Path:
        return Path(__file__).resolve().parents[3]

    def _structured_error(
        self,
        *,
        code: str,
        message: str,
        extra: dict[str, JsonValue] | None = None,
        accepted: bool | None = None,
        finished: bool = False,
    ) -> ToolOutput:
        error_payload: dict[str, JsonValue] = {"code": code}
        if extra is not None:
            error_payload.update(extra)
        metadata: dict[str, JsonValue] = {"error": error_payload}
        if accepted is not None:
            metadata["accepted"] = accepted
        return _tool_output(
            message,
            metadata=metadata,
            detail={"error": error_payload},
            finished=finished,
        )


__all__ = [
    "EpisodeMetricsSnapshot",
    "ToolCallFailure",
    "TradeBenchSessionRuntime",
]
