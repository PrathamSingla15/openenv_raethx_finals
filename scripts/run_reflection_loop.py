"""Run a baseline rollout then N reflection iterations and plot rewards.

Workflow:

  1. Run a baseline rollout (iter 0) using the in-tree ``inference.SYSTEM_PROMPT``.
  2. For each reflection iteration ``i`` in ``1..N``:
     - Read the prior run's artifacts.
     - Build a GEPA-style meta-prompt and call Claude Opus 4.7
       (via OpenRouter) to propose an improved system prompt.
     - Re-run the rollout with the new prompt.
  3. Save every iteration's run dir, the proposed prompts, and the score
     history under ``artifacts/reflection_<ts>/``.
  4. Render a reward-vs-iteration plot (PNG) at the parent dir.

Environment:

  HF_TOKEN              required - Hugging Face Inference Provider key
                        (drives the Qwen3-32B/Groq rollouts)
  OPENROUTER_API_KEY    required - OpenRouter key (drives the Opus 4.7
                        reflection calls)
  ENV_URL               required - URL of the running TradeBench server
                        (e.g. http://127.0.0.1:8773)
  MODEL_NAME            optional - rollout model id (default
                        ``Qwen/Qwen3-32B:groq``)
  REFLECTION_MODEL      optional - reflector model id (default
                        ``anthropic/claude-opus-4.7``)
  MAX_STEPS_PER_TASK    optional - cap on env steps per rollout
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import inference  # type: ignore[import-not-found]
import reflection  # type: ignore[import-not-found]


def _ts() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


async def _run_one_rollout(
    *,
    iteration: int,
    system_prompt: str,
    suffix: str,
) -> dict[str, Any]:
    print(f"\n{'#' * 60}", flush=True)
    print(f"# ROLLOUT iter={iteration} suffix={suffix}", flush=True)
    print(f"{'#' * 60}", flush=True)
    result = await inference.main(system_prompt=system_prompt, run_id_suffix=suffix)
    return result


def _save_proposed_prompt(
    *,
    parent_dir: Path,
    iteration: int,
    new_prompt: str,
    raw_response: str,
    meta_prompt: str,
    reflector_system_prompt: str,
) -> None:
    iter_dir = parent_dir / f"iter_{iteration:02d}__reflection"
    iter_dir.mkdir(parents=True, exist_ok=True)
    (iter_dir / "new_system_prompt.txt").write_text(new_prompt, encoding="utf-8")
    (iter_dir / "reflector_raw_response.txt").write_text(raw_response, encoding="utf-8")
    (iter_dir / "meta_prompt.txt").write_text(meta_prompt, encoding="utf-8")
    (iter_dir / "reflector_system_prompt.txt").write_text(
        reflector_system_prompt,
        encoding="utf-8",
    )


def _plot_rewards(history: list[dict[str, Any]], out_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "[WARN] matplotlib not available; skipping plot",
            file=sys.stderr,
            flush=True,
        )
        return

    iters = [row["iteration"] for row in history]
    test_scores = [row["test_score"] for row in history]
    cumulative = [row["test_cumulative_reward"] for row in history]
    normalized = [row.get("test_score_normalized", 0.5) for row in history]

    fig, axes = plt.subplots(3, 1, figsize=(8.5, 9.5), sharex=True)
    ax_top, ax_mid, ax_bot = axes

    ax_top.plot(iters, normalized, marker="o", linewidth=2.2, color="#1565c0")
    ax_top.axhline(0.5, color="#888", linewidth=0.8, linestyle="--", label="neutral (0.5)")
    if normalized:
        ax_top.axhline(
            normalized[0],
            color="#d32f2f",
            linewidth=0.8,
            linestyle=":",
            label=f"baseline = {normalized[0]:.3f}",
        )
    ax_top.set_ylim(0.0, 1.0)
    ax_top.set_ylabel("score_normalized in [0, 1]\nsigmoid(cumulative_reward)")
    ax_top.set_title(
        "Reflective prompt optimisation - reward vs iteration (0 = baseline)",
    )
    ax_top.legend(loc="best", fontsize=9)
    ax_top.grid(alpha=0.3)

    ax_mid.plot(iters, cumulative, marker="s", linewidth=2, color="#2e7d32")
    ax_mid.axhline(0, color="#888", linewidth=0.8, linestyle="--")
    if cumulative:
        ax_mid.axhline(
            cumulative[0],
            color="#d32f2f",
            linewidth=0.8,
            linestyle=":",
            label=f"baseline = {cumulative[0]:+.4f}",
        )
        ax_mid.legend(loc="best", fontsize=9)
    ax_mid.set_ylabel("test cumulative composite reward (raw)")
    ax_mid.grid(alpha=0.3)

    ax_bot.plot(iters, test_scores, marker="^", linewidth=2, color="#6a1b9a")
    ax_bot.axhline(0, color="#888", linewidth=0.8, linestyle="--")
    if test_scores:
        ax_bot.axhline(
            test_scores[0],
            color="#d32f2f",
            linewidth=0.8,
            linestyle=":",
            label=f"baseline = {test_scores[0]:+.4f}",
        )
        ax_bot.legend(loc="best", fontsize=9)
    ax_bot.set_xlabel("iteration")
    ax_bot.set_ylabel("test ROI (portfolio_value / initial - 1)")
    ax_bot.grid(alpha=0.3)
    ax_bot.set_xticks(iters)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


async def run_reflection_loop(
    *,
    n_iterations: int,
    parent_dir: Path,
    task_ids: list[str],
    rollout_model: str | None,
    from_run: Path | None = None,
    initial_prompt_file: Path | None = None,
) -> list[dict[str, Any]]:
    parent_dir.mkdir(parents=True, exist_ok=True)
    inference.TASK_IDS = task_ids
    if rollout_model:
        os.environ["MODEL_NAME"] = rollout_model

    history: list[dict[str, Any]] = []
    if initial_prompt_file is not None:
        current_prompt = initial_prompt_file.read_text(encoding="utf-8")
        print(
            f"\n[CONTINUE] Using prompt from {initial_prompt_file} as iter 0 baseline "
            f"({len(current_prompt)} chars)",
            flush=True,
        )
    else:
        current_prompt = inference.SYSTEM_PROMPT
    (parent_dir / "iter_00__baseline").mkdir(parents=True, exist_ok=True)
    (parent_dir / "iter_00__baseline" / "system_prompt.txt").write_text(
        current_prompt,
        encoding="utf-8",
    )

    if from_run is not None:
        from_run = from_run.resolve()
        print(
            f"\n[REUSE] Skipping baseline rollout; using existing run as iter 0:\n"
            f"  {from_run}",
            flush=True,
        )
        baseline = _baseline_from_existing_run(from_run)
    else:
        baseline = await _run_one_rollout(
            iteration=0,
            system_prompt=current_prompt,
            suffix="iter00_baseline",
        )
    history.append(_history_row(0, baseline, prior_run=None))
    _write_history(parent_dir, history)

    for it in range(1, n_iterations + 1):
        prior = history[-1]
        prior_artifacts = prior.get("artifacts_root")
        if not prior_artifacts:
            print(
                f"[WARN] prior iteration {it - 1} has no artifacts_root; aborting reflection loop",
                file=sys.stderr,
                flush=True,
            )
            break
        ctx = reflection.load_run_context(
            Path(prior_artifacts),
            iteration=it,
            history_snapshot=list(history),
        )
        client = reflection.make_openrouter_client()
        new_prompt, raw_response, reflector_system_prompt, meta_prompt = (
            reflection.propose_improved_prompt(ctx, client=client)
        )
        print(
            f"\n[REFLECTION] iter={it} meta_prompt={len(meta_prompt)} chars  "
            f"new_prompt={len(new_prompt)} chars  budget={ctx.target_prompt_chars}",
            flush=True,
        )
        if not new_prompt or len(new_prompt) < 200:
            print(
                f"[WARN] reflector returned a too-short prompt ({len(new_prompt)} chars); "
                "keeping prior prompt for this iteration",
                file=sys.stderr,
                flush=True,
            )
            new_prompt = current_prompt
        _save_proposed_prompt(
            parent_dir=parent_dir,
            iteration=it,
            new_prompt=new_prompt,
            raw_response=raw_response,
            meta_prompt=meta_prompt,
            reflector_system_prompt=reflector_system_prompt,
        )
        current_prompt = new_prompt
        result = await _run_one_rollout(
            iteration=it,
            system_prompt=new_prompt,
            suffix=f"iter{it:02d}_reflect",
        )
        history.append(_history_row(it, result, prior_run=prior_artifacts))
        _write_history(parent_dir, history)

    plot_path = parent_dir / "scores.png"
    _plot_rewards(history, plot_path)
    print(f"\n[DONE] history -> {parent_dir / 'history.json'}", flush=True)
    print(f"[DONE] plot    -> {plot_path}", flush=True)
    return history


def _baseline_from_existing_run(run_dir: Path) -> dict[str, Any]:
    """Reconstruct an inference.main()-style result dict from a run on disk.

    Used when ``--from-run`` is supplied so the reflection loop can treat
    an already-completed rollout as iter 0 instead of re-running it.
    """

    test_summary_path = run_dir / "test" / "summary.json"
    manifest_path = run_dir / "manifest.json"
    test_summary = json.loads(test_summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cum = float(test_summary.get("cumulative_reward", 0.0))
    bars = int(test_summary.get("bars_completed", 0))
    score_norm = float(test_summary.get("score_normalized", 0.5))
    final_pv = float(test_summary.get("final_portfolio_value", 0.0))
    initial_pv = float(test_summary.get("initial_portfolio_value", 1.0)) or 1.0
    return {
        "run_id": manifest.get("run_id", run_dir.name),
        "artifacts_root": str(run_dir.resolve()),
        "scores": dict(manifest.get("scores", {})),
        "test_score": (final_pv / initial_pv) - 1.0 if initial_pv > 0 else 0.0,
        "test_cumulative_reward": cum,
        "test_bars_completed": bars,
        "test_score_normalized": score_norm,
        "test_action_counts": dict(manifest.get("action_counts", {}).get("test", {})),
    }


def _history_row(
    iteration: int,
    result: dict[str, Any],
    *,
    prior_run: str | None,
) -> dict[str, Any]:
    return {
        "iteration": iteration,
        "run_id": result.get("run_id"),
        "artifacts_root": result.get("artifacts_root"),
        "test_score": float(result.get("test_score", 0.0)),
        "test_cumulative_reward": float(result.get("test_cumulative_reward", 0.0)),
        "test_bars_completed": int(result.get("test_bars_completed", 0)),
        "test_score_normalized": float(result.get("test_score_normalized", 0.5)),
        "test_action_counts": dict(result.get("test_action_counts", {})),
        "scores": dict(result.get("scores", {})),
        "prior_run": prior_run,
    }


def _write_history(parent_dir: Path, history: list[dict[str, Any]]) -> None:
    (parent_dir / "history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=5, help="Number of reflection iterations after the baseline")
    parser.add_argument(
        "--task-ids",
        nargs="+",
        default=["train", "test"],
        help="Tier sequence to run each iteration (default: train test)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Parent dir under artifacts/ (default artifacts/reflection_<ts>/)",
    )
    parser.add_argument(
        "--rollout-model",
        default=None,
        help="Override MODEL_NAME for the rollout (e.g. Qwen/Qwen3-32B:groq)",
    )
    parser.add_argument(
        "--from-run",
        type=Path,
        default=None,
        help=(
            "Use an existing run directory as iter 0 instead of running a "
            "fresh baseline. Path should point at artifacts/runs/<run_id>/."
        ),
    )
    parser.add_argument(
        "--initial-prompt-file",
        type=Path,
        default=None,
        help=(
            "Override iter-0 baseline prompt with the contents of this file. "
            "Pair with --from-run to continue from a prior reflection's "
            "final iter (use that iter's new_system_prompt.txt and run dir)."
        ),
    )
    args = parser.parse_args()

    artifacts_root = Path(
        os.environ.get(
            "TRADEBENCH_ARTIFACTS_ROOT",
            str(PROJECT_ROOT / "artifacts"),
        ),
    )
    parent_dir = args.out or (artifacts_root / f"reflection_{_ts()}")

    if "OPENROUTER_API_KEY" not in os.environ:
        print(
            "[ERROR] OPENROUTER_API_KEY env var is required (Opus 4.7 reflection calls).",
            file=sys.stderr,
            flush=True,
        )
        return 2

    history = asyncio.run(
        run_reflection_loop(
            n_iterations=args.iters,
            parent_dir=parent_dir,
            task_ids=list(args.task_ids),
            rollout_model=args.rollout_model,
            from_run=args.from_run,
            initial_prompt_file=args.initial_prompt_file,
        ),
    )

    print(f"\n{'=' * 60}", flush=True)
    print("REFLECTION HISTORY", flush=True)
    print(f"{'=' * 60}", flush=True)
    for row in history:
        tag = "baseline" if row["iteration"] == 0 else f"iter {row['iteration']:02d}"
        print(
            f"  {tag:8s}  test_score={row['test_score']:+.4f}  "
            f"cum_reward={row['test_cumulative_reward']:+.4f}  "
            f"bars={row['test_bars_completed']}  run_id={row['run_id']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
