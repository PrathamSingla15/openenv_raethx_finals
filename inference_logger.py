"""Per-rollout artifact writer for the inference driver.

A ``RunLogger`` instance owns one ``artifacts/runs/<run_id>/`` directory
and streams every LLM call, env.step, and phase-level summary into it so
a downstream reflection driver can read back what happened bar-by-bar
without rerunning the rollout.

Layout produced::

    artifacts/runs/<run_id>/
        manifest.json            run-level: model, tier_pair, scores, action_counts, timing
        system_prompt.txt        snapshot of SYSTEM_PROMPT used
        carry_strategy.txt       text the test phase received from train
        train/
            study_packet.txt     full in-context packet shown to the model
            llm_calls.jsonl      one row per LLM call (prompt, raw, parsed, warning)
            trajectory.jsonl     one row per env.step
            summary.json         attempts, used_fallback, regime_label, edge_summary
        test/
            llm_calls.jsonl
            trajectory.jsonl
            summary.json         bars_completed, total_steps, final_score,
                                 cumulative_reward, regime_labels, violations
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO


def make_run_id(model: str, tier_pair: str) -> str:
    """Compose a sortable, filesystem-safe run id from current time + slugs."""

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    model_slug = re.sub(r"[^A-Za-z0-9]+", "-", model).strip("-").lower() or "model"
    tier_slug = re.sub(r"[^A-Za-z0-9]+", "-", tier_pair).strip("-").lower() or "tiers"
    return f"{ts}__{tier_slug}__{model_slug}"


class RunLogger:
    """Streaming JSONL writer for a single inference rollout.

    The logger is phase-aware: ``begin_phase("train")`` opens trajectory and
    LLM-call streams under ``<root>/train/``; calling it again with another
    name closes the prior streams and opens fresh ones for that phase. All
    file handles are flushed on each write, so a crashed driver still
    leaves a partially-readable trace on disk.
    """

    def __init__(
        self,
        artifacts_root: Path,
        run_id: str,
        *,
        model: str,
        tier_pair: str,
    ) -> None:
        self.run_id = run_id
        self.model = model
        self.tier_pair = tier_pair
        self.root = (artifacts_root / "runs" / run_id).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._phase: str | None = None
        self._phase_dir: Path | None = None
        self._traj_handle: TextIO | None = None
        self._llm_handle: TextIO | None = None
        self._scores: dict[str, float] = {}
        self._action_counts: dict[str, dict[str, int]] = {}
        self._llm_call_counts: dict[str, int] = {}
        self._started_at = datetime.now(UTC).isoformat()
        self._notes: list[str] = []

    def write_system_prompt(self, prompt: str) -> None:
        (self.root / "system_prompt.txt").write_text(prompt, encoding="utf-8")

    def write_carry_strategy(self, strategy: str) -> None:
        (self.root / "carry_strategy.txt").write_text(strategy, encoding="utf-8")

    def begin_phase(self, phase: str) -> None:
        self._close_phase_handles()
        self._phase = phase
        self._phase_dir = self.root / phase
        self._phase_dir.mkdir(parents=True, exist_ok=True)
        self._traj_handle = (self._phase_dir / "trajectory.jsonl").open(
            "w",
            encoding="utf-8",
        )
        self._llm_handle = (self._phase_dir / "llm_calls.jsonl").open(
            "w",
            encoding="utf-8",
        )
        self._action_counts.setdefault(phase, {})
        self._llm_call_counts.setdefault(phase, 0)

    def write_study_packet(self, text: str) -> None:
        if self._phase_dir is None:
            return
        (self._phase_dir / "study_packet.txt").write_text(text, encoding="utf-8")

    def log_llm_call(
        self,
        *,
        step: int,
        prompt_user: str,
        raw_response: str,
        parsed_action: dict[str, Any],
        parse_warning: str = "",
    ) -> None:
        if self._llm_handle is None:
            return
        record = {
            "step": step,
            "ts": datetime.now(UTC).isoformat(),
            "prompt_user": prompt_user,
            "raw_response": raw_response,
            "parsed_action": parsed_action,
            "parse_warning": parse_warning,
        }
        self._llm_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._llm_handle.flush()
        if self._phase is not None:
            self._llm_call_counts[self._phase] = (
                self._llm_call_counts.get(self._phase, 0) + 1
            )

    def log_step(
        self,
        *,
        step: int,
        action_type: str,
        payload: dict[str, Any],
        observation: Any,
        reward: float,
    ) -> None:
        if self._traj_handle is None:
            return
        record = {
            "step": step,
            "ts": datetime.now(UTC).isoformat(),
            "action_type": action_type,
            "payload": payload,
            "reward": reward,
            "done": bool(getattr(observation, "done", False)),
            "error": getattr(observation, "error", None),
            "current_date": getattr(observation, "current_date", ""),
            "bars_remaining": int(getattr(observation, "bars_remaining", 0) or 0),
            "portfolio_value": float(getattr(observation, "portfolio_value", 0.0) or 0.0),
            "cash": float(getattr(observation, "cash", 0.0) or 0.0),
            "positions": dict(getattr(observation, "positions", {}) or {}),
            "reward_breakdown": dict(getattr(observation, "reward_breakdown", {}) or {}),
            "violations": list(getattr(observation, "violations", []) or []),
            "tool_output_excerpt": _truncate(
                getattr(observation, "tool_output", "") or "",
                limit=2000,
            ),
        }
        self._traj_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._traj_handle.flush()
        if self._phase is not None:
            counts = self._action_counts.setdefault(self._phase, {})
            counts[action_type] = counts.get(action_type, 0) + 1

    def write_phase_summary(self, summary: dict[str, Any]) -> None:
        if self._phase_dir is None:
            return
        (self._phase_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def record_score(self, phase: str, score: float) -> None:
        self._scores[phase] = float(score)

    def add_note(self, note: str) -> None:
        self._notes.append(str(note))

    def close(self) -> None:
        self._close_phase_handles()
        manifest = {
            "run_id": self.run_id,
            "started_at": self._started_at,
            "ended_at": datetime.now(UTC).isoformat(),
            "model": self.model,
            "tier_pair": self.tier_pair,
            "scores": self._scores,
            "action_counts": self._action_counts,
            "llm_call_counts": self._llm_call_counts,
            "notes": self._notes,
        }
        (self.root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _close_phase_handles(self) -> None:
        if self._traj_handle is not None:
            self._traj_handle.close()
            self._traj_handle = None
        if self._llm_handle is not None:
            self._llm_handle.close()
            self._llm_handle = None


def _truncate(text: str, *, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"... [truncated {len(text) - limit} chars]"


__all__ = ["RunLogger", "make_run_id"]
