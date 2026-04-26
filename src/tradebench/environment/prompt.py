"""Prompt builder for the TradeBench environment."""

from __future__ import annotations

from datetime import date

from tradebench.environment._io import TextBlock
from tradebench.episodes.models import EpisodeManifest
from tradebench.sandbox.models import EpisodeWorkspace


# Appended to every episode prompt. Violations trigger a -1.0 r_rules penalty.
RULES_BASED_STRATEGY_CLAUSE = "\n".join(
    [
        "## RULES-BASED STRATEGY REQUIREMENT (non-optional)",
        (
            "Base every decision on code-expressible, systematic models that use "
            "ONLY the market data surfaced to you via the PIT query service up to "
            "the current simulation date."
        ),
        "",
        "Do NOT:",
        "- Recall specific historical events or outcomes from your training data",
        "- Reference \"what actually happened\" in any period",
        "- Use ticker-date pairs from memory (e.g., \"I know AAPL rallied in 2023\")",
        "",
        (
            "Violation will be detected and flagged. A single violation triggers a "
            "large negative reward penalty and may terminate the episode. The goal "
            "is systematic strategy derivation from explicit data, not retrieval "
            "of memorized market history."
        ),
    ],
)


# Appended after RULES_BASED_STRATEGY_CLAUSE on the train tier.
SELF_VALIDATION_CLAUSE = "\n".join(
    [
        "## SELF-VALIDATION (there is no separate validation set)",
        (
            "This environment has only two tiers you should care about: the "
            "training tier you are in now, and a held-out test tier you will "
            "never see during training. There is no validation split between "
            "them."
        ),
        "",
        (
            "If you want to validate a hypothesis before betting size on it, "
            "do it INSIDE the sandbox using only the bars whose session_date "
            "≤ the current simulation date. The PIT query layer enforces this "
            "automatically — any future-dated query is rejected — so you can "
            "safely:"
        ),
        "",
        "- Pick a sub-window of the past N visible bars (e.g. the trailing 40 days).",
        "- Split that sub-window into a train half and an eval half.",
        "- Fit your model on the train half, score it on the eval half.",
        "- Use the eval-half score to decide whether to act on the model today.",
        "",
        (
            "Do NOT treat the full visible history as both train and validation "
            "— that is in-sample evaluation and will overstate your edge. Reserve "
            "an eval slice and only score against it."
        ),
    ],
)


def build_prompt(
    *,
    manifest: EpisodeManifest,
    current_date: date,
    next_session_date: date | None,
    workspace: EpisodeWorkspace,
    visible_universe_count: int,
) -> list[TextBlock]:
    next_session = (
        next_session_date.isoformat() if next_session_date is not None else "none"
    )
    sections: list[str] = [
        "You are operating inside a deterministic TradeBench trading episode.",
        "",
        f"Task ID: {manifest.task_id}",
        f"Dataset version: {manifest.dataset_version}",
        f"Tier role: {manifest.split.name}",
        f"Current decision date: {current_date.isoformat()}",
        f"Next execution session: {next_session}",
        (
            "Orders placed today execute at the next session open. "
            "Before the first order on any decision date, you must call "
            "`record_decision`."
        ),
        "",
        "Episode window:",
        f"- Warmup start: {manifest.warmup_start.isoformat()}",
        f"- Live start: {manifest.episode_start.isoformat()}",
        f"- Live end: {manifest.episode_end.isoformat()}",
        f"- Initial cash: {manifest.initial_cash}",
        f"- Visible universe size today: {visible_universe_count}",
        "",
        "Sandbox workspace:",
        f"- Work dir: {workspace.container_work_dir}",
        f"- Read-only data dir: {workspace.container_data_dir}",
        f"- Read-only meta dir: {workspace.container_meta_dir}",
        f"- Output dir: {workspace.container_output_dir}",
        "",
        "Code execution:",
        "You can run arbitrary Python or shell commands inside the sandbox using",
        "the `sandbox_exec` tool. The sandbox has read-only access to market data",
        f"at {workspace.container_data_dir} and a read-write work directory at",
        f"{workspace.container_work_dir}. Use this to build models, run analyses,",
        "or compute signals before making trading decisions.",
        "",
        (
            "Use environment tools to inspect state, submit orders, cancel "
            "queued orders, "
        ),
        "advance the market clock, run code in the sandbox, and inspect metrics.",
        "",
        RULES_BASED_STRATEGY_CLAUSE,
    ]
    if manifest.split.name == "train":
        sections.extend(["", SELF_VALIDATION_CLAUSE])
    return [TextBlock(text="\n".join(sections))]


__all__ = ["RULES_BASED_STRATEGY_CLAUSE", "SELF_VALIDATION_CLAUSE", "build_prompt"]
