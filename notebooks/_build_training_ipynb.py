"""Generator for notebooks/training.ipynb. Run once when content changes:

    python notebooks/_build_training_ipynb.py

Keeps the canonical notebook source in one readable Python file, with cell
content as ordinary triple-quoted strings, and emits a clean .ipynb.
"""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

OUT = Path(__file__).parent / "training.ipynb"


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(text).strip().splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(text).strip().splitlines(keepends=True),
    }


CELLS = [
    md("""
        # TradeBench: Reflection-loop training notebook

        [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PrathamSingla15/openenv_raethx_finals/blob/main/notebooks/training.ipynb)

        End-to-end runnable training notebook for the TradeBench reflection-based prompt optimization loop. Pairs with the [code repository](https://github.com/PrathamSingla15/openenv_raethx_finals) and the [deployed environment](https://huggingface.co/spaces/yobro4619/tradebench).

        **What this notebook does**

        1. Clones the repo and installs dependencies.
        2. Configures API keys (HuggingFace inference, OpenRouter for the reflector).
        3. Verifies the deployed env is reachable.
        4. Walks through the reflection meta-prompt and the 1.10x length cap, so you can see exactly what the loop does.
        5. Runs the reflection loop end-to-end. The default is 1 iteration so the notebook completes in a single Colab session; bumping `NUM_ITERATIONS` to 5 reproduces the headline result.
        6. Loads the resulting `history.json` and plots the score trajectory.
        7. Diffs the baseline system prompt against the final iter's prompt, so you can read what changed.

        **What it's training**

        The model weights are not touched. Across iterations, the only artifact that changes is the agent's *system prompt*. Claude Opus 4.7 reads the prior trajectory, identifies the single most-costly failure mode, and emits a surgical edit. The model running the agent rollouts (Qwen3-32B by default, GLM-5.1 also tested) stays exactly as the inference provider serves it.

        Headline result on Qwen3-32B: `score_normalized` 0.6155 → 0.6584 over 5 iters, ROI +2.78% → +9.89%, monotone climb.
    """),

    md("""
        ## 1. Clone the repository
    """),
    code("""
        !git clone https://github.com/PrathamSingla15/openenv_raethx_finals.git
        %cd openenv_raethx_finals
    """),

    md("""
        ## 2. Install dependencies

        We use `uv` for fast, reproducible installs in local development. Inside Colab, plain `pip` against the project's `pyproject.toml` is simpler.
    """),
    code("""
        !pip install -q --upgrade pip
        !pip install -q -e ".[openai,plots]"
    """),

    md("""
        ## 3. Configure API keys

        Two services are needed:

        - **HF_TOKEN**, used to call the agent rollout model through HuggingFace's Inference Providers (Groq for `qwen/qwen3-32b:groq`, Together for `zai-org/GLM-5.1:together`).
        - **OPENROUTER_API_KEY**, used to call Claude Opus 4.7 as the reflector.

        Both are read at runtime from environment variables. We use `getpass` so the keys do not appear in cell output.
    """),
    code("""
        import getpass
        import os

        if not os.environ.get("HF_TOKEN"):
            os.environ["HF_TOKEN"] = getpass.getpass("HF_TOKEN: ")

        if not os.environ.get("OPENROUTER_API_KEY"):
            os.environ["OPENROUTER_API_KEY"] = getpass.getpass("OPENROUTER_API_KEY: ")
    """),

    md("""
        ## 4. Point at the deployed environment

        The TradeBench env is hosted as a Hugging Face Space. We talk to it over HTTP/WebSocket from this notebook, so no local Docker is required. Verify it is awake before launching the loop.
    """),
    code("""
        import urllib.request
        import json

        ENV_URL = "https://yobro4619-tradebench.hf.space"
        os.environ["ENV_URL"] = ENV_URL

        with urllib.request.urlopen(f"{ENV_URL}/health", timeout=10) as resp:
            health = json.loads(resp.read())
        print(json.dumps(health, indent=2))
        assert health.get("status") == "healthy", "Env is not healthy"
    """),

    md("""
        ## 5. The reflection loop in concept

        Each iteration is one rollout plus one reflector call. The rollout drives the agent through the train and test phases of the env, recording every action to `artifacts/runs/<ts>__<model>__iterNN_reflect/`. The reflector reads that trajectory and emits a new system prompt for the next iteration.

        Two structural choices keep the loop honest:

        1. **Surgical-edit constraint.** The new prompt's length must be `<= 1.10 * prior_length`. This stops the reflector from rewriting from scratch and forces edits to stay local.
        2. **Single-failure-mode rule.** The reflector is told to identify the *one* most-costly failure mode in the trajectory and fix only that. One thing per iteration.

        Below is the meta-prompt template we send to the reflector. Reading it once explains what the loop is actually optimizing.
    """),
    code("""
        from pathlib import Path

        # Show the surgical-edit-constraint helper. This is what enforces the 1.10x
        # length cap and the single-failure-mode rule that keep the loop honest.
        reflection_src = Path("reflection.py").read_text()
        for chunk in reflection_src.split("\\n\\n"):
            if "1.10" in chunk or "length_cap" in chunk or "STEP 1" in chunk:
                print(chunk)
                print("---")
    """),

    md("""
        ## 6. Run the reflection loop

        Set `NUM_ITERATIONS` to control how many reflection passes to do. The default of `1` lets the notebook complete inside a single Colab session (~30-50 minutes). Set it to `5` to reproduce the headline result.

        The loop calls `scripts/run_reflection_loop.py`, which orchestrates rollout-then-reflect, writes per-iter artifacts under `artifacts/runs/`, and updates `artifacts/reflection_<ts>/history.json` after each iteration.
    """),
    code("""
        NUM_ITERATIONS = 1  # bump to 5 to reproduce the full trajectory
        ROLLOUT_MODEL = "qwen/qwen3-32b:groq"  # alternatives: "zai-org/GLM-5.1:together"

        os.environ["MODEL_NAME"] = ROLLOUT_MODEL

        !uv run python scripts/run_reflection_loop.py \\
            --iters {NUM_ITERATIONS} \\
            --rollout-model {ROLLOUT_MODEL}
    """),

    md("""
        ## 7. Inspect the score trajectory

        After the loop finishes, `artifacts/reflection_<ts>/history.json` records one row per iteration. Each row has `iteration`, `run_id`, `test_score_normalized`, `test_action_counts`, and the corresponding artifact paths.
    """),
    code("""
        import json
        from pathlib import Path

        reflection_roots = sorted(Path("artifacts").glob("reflection_*"), reverse=True)
        latest = reflection_roots[0]
        print(f"Latest reflection root: {latest}")

        history = json.loads((latest / "history.json").read_text())
        for row in history:
            ac = row.get("test_action_counts") or {}
            roi_pct = (row.get("test_score") or 0.0) * 100
            print(
                f"iter {row['iteration']}: "
                f"score={row.get('test_score_normalized'):.4f}  "
                f"ROI={roi_pct:+.2f}%  "
                f"place_order={ac.get('place_order', '?')}  "
                f"sandbox_exec={ac.get('sandbox_exec', '?')}"
            )
    """),

    md("""
        ## 8. Plot the score evolution

        Quick visualization of `score_normalized` across iterations. The repository ships a more thorough multi-model plotter at `scripts/visualize/render_all.py`; this cell is a one-liner you can run inline.
    """),
    code("""
        import matplotlib.pyplot as plt

        xs = [row["iteration"] for row in history]
        ys = [row["test_score_normalized"] for row in history]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(xs, ys, "-o", linewidth=2.0, markersize=7, color="#1f77b4")
        ax.set_xlabel("Reflection iteration")
        ax.set_ylabel("score_normalized (test phase)")
        ax.set_title("Reflection-loop reward evolution")
        ax.grid(True, alpha=0.3)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=9)
        plt.tight_layout()
        plt.show()
    """),

    md("""
        ## 9. Read the prompt diff

        The whole point of reflection-based optimization is that the optimization trace is human-readable. After running, you can diff the baseline system prompt against the final iteration's prompt and see exactly what changed.
    """),
    code("""
        import difflib

        baseline_run = Path(history[0]["artifacts_root"])
        baseline_prompt = (baseline_run / "system_prompt.txt").read_text()

        last_iter_dirs = sorted(latest.glob("iter_*__reflection"))
        if last_iter_dirs:
            final_prompt = (last_iter_dirs[-1] / "new_system_prompt.txt").read_text()
            diff = difflib.unified_diff(
                baseline_prompt.splitlines(keepends=True),
                final_prompt.splitlines(keepends=True),
                fromfile="iter_0/system_prompt.txt",
                tofile=f"iter_{history[-1]['iteration']}/new_system_prompt.txt",
                n=2,
            )
            print("".join(diff)[:6000])  # first 6KB of diff
        else:
            print("No reflection iters yet. Run the loop with NUM_ITERATIONS >= 1.")
    """),

    md("""
        ## 10. Where to go from here

        - `RESULTS.md` walks through the full 5-iteration result on Qwen3-32B and the GLM-5.1 comparison run.
        - `Blog.md` is the long-form technical writeup, including the methodology justification (why reflection rather than RL given a 36-hour hackathon budget and 45-50 minute rollouts).
        - `docs/figures/` holds publication-quality plots of every metric used in the writeup.
        - `scripts/visualize/render_all.py` regenerates all six figures from any history.json.

        To run the loop yourself, the minimum is:

        ```bash
        ENV_URL="https://yobro4619-tradebench.hf.space" \\
        HF_TOKEN="..." \\
        OPENROUTER_API_KEY="..." \\
        uv run python scripts/run_reflection_loop.py --iters 5 --rollout-model qwen/qwen3-32b:groq
        ```

        That is the same command this notebook ran in cell 6.
    """),
]

NOTEBOOK = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.12",
        },
        "colab": {"provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def main() -> None:
    OUT.write_text(json.dumps(NOTEBOOK, indent=1) + "\n")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes, {len(CELLS)} cells)")


if __name__ == "__main__":
    main()
