"""Baselines tab — three reference strategies, the trained-vs-baseline plot."""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr


_HERE = Path(__file__).resolve().parents[3]
_RUNS_DIR = _HERE / "notebooks" / "runs"
_PLOTS_DIR = _HERE / "docs" / "plots"


_INTRO_HTML = """
<div class="tb-section-eyebrow">three reference strategies</div>
<h2 class="tb-section-title">Baselines establish the floor for the
"trained vs untrained" comparison the hackathon brief explicitly rewards.</h2>

<p class="tb-section-lead">
All three are deterministic, async coroutines under
<code>tradebench.baselines</code>. Drive them via
<code>uv run python inference.py --baseline {cash,equal_weight,kelly}</code>;
each emits the same <code>[START]</code> / <code>[STEP]</code> /
<code>[END]</code> log shape as the LLM driver, so the auto-validator parses
either output identically.
</p>

<table class="tb-table">
    <thead>
        <tr><th>Baseline</th><th>Exposure</th><th>Allocation</th><th>Rebalance</th></tr>
    </thead>
    <tbody>
        <tr><td><strong>cash</strong></td><td class="mono">0%</td><td>—</td><td>—</td></tr>
        <tr><td><strong>equal_weight</strong></td><td class="mono">90%</td><td>1/N across the visible universe</td><td>One-shot at episode start</td></tr>
        <tr><td><strong>kelly</strong></td><td class="mono">50%</td><td>1/N × half-Kelly fraction</td><td>One-shot at episode start</td></tr>
    </tbody>
</table>
"""


def _load_run_summary() -> dict:
    """Read the most recent run summaries on disk; returns ``{}`` when absent."""

    summary: dict = {}
    grpo_loss = _RUNS_DIR / "grpo_loss.json"
    if grpo_loss.is_file():
        try:
            summary["grpo_loss"] = json.loads(grpo_loss.read_text()).get("loss", [])
        except json.JSONDecodeError:
            pass
    gepa_history = _RUNS_DIR / "gepa_history.json"
    if gepa_history.is_file():
        try:
            summary["gepa_history"] = json.loads(gepa_history.read_text())
        except json.JSONDecodeError:
            pass
    return summary


def _ribbon_html() -> str:
    """Embed baseline_vs_trained.png inline as base64.

    Gradio's ``/gradio_api/file=`` route requires ``allowed_paths`` registration
    at mount time and HF Spaces sandboxing varies; base64-embedding sidesteps
    both so the PNG ships in the rendered HTML bytes.
    """

    target = _PLOTS_DIR / "baseline_vs_trained.png"
    if target.is_file():
        import base64

        b64 = base64.b64encode(target.read_bytes()).decode("ascii")
        return f"""
<div class="tb-ribbon">
    <img src="data:image/png;base64,{b64}"
         alt="cumulative wealth — baselines vs trained"
         style="width: 100%; height: auto; border-radius: 4px;"/>
</div>
"""
    return """
<div class="tb-ribbon">
    <p>Cumulative-wealth plot pending. Run
    <code>uv run python scripts/generate_plots.py --tier t1 --max-bars 60</code>
    to populate <code>docs/plots/baseline_vs_trained.png</code>.</p>
</div>
"""


def _run_summary_html() -> str:
    summary = _load_run_summary()
    if not summary:
        return """
<p class="tb-section-lead tb-section-lead-mute">
No run summaries on disk under <code>notebooks/runs/</code> yet. Rollouts run
after submission populate this section. The Path A scaffolding (TRL + Unsloth
+ Qwen3-1.7B GRPO dry-run) emits <code>grpo_loss.json</code>; the Path B
scaffolding (GEPA prompt evolution) emits <code>gepa_history.json</code>.
</p>
"""

    bits: list[str] = []
    if "grpo_loss" in summary:
        n = len(summary["grpo_loss"])
        first, last = summary["grpo_loss"][0], summary["grpo_loss"][-1]
        bits.append(
            f"<li><strong>Path A · GRPO dry-run</strong> — "
            f"{n} steps; loss {first:.3f} → {last:.3f}</li>",
        )
    if "gepa_history" in summary:
        gens = summary["gepa_history"]
        bits.append(
            f"<li><strong>Path B · GEPA prompt evolution</strong> — "
            f"{len(gens)} generation(s) logged</li>",
        )
    return "<ul style='line-height: 1.8;'>" + "".join(bits) + "</ul>"


_PATH_HTML = """
<div class="tb-section-eyebrow">improvement evidence · two paths</div>
<h2 class="tb-section-title">Path B is the headline; Path A ships as RL-readiness proof.</h2>

<p class="tb-section-lead">
The brief asks for a "Working training script using Unsloth or HF TRL" — Path A
provides that literally. The improvement-evidence story is GEPA-style prompt
evolution against TradeBench's verifiable composite reward (Path B); the
real-derived (aliased + date-shifted) tier data carries genuine fat tails,
correlation structure, and regime shifts for the model to actually learn from.
</p>

<ul style="line-height: 1.8; max-width: 62ch;">
    <li><code>notebooks/training_grpo.py</code> — TRL <code>GRPOTrainer</code> +
        Unsloth Qwen3-1.7B 4-bit LoRA, 7 reward functions plumbed (one per
        <code>RewardBreakdown</code> field). Dry-run runs end-to-end on CPU; the
        <code># ONSITE</code>-tagged cells replace the synthetic loss series with a
        real training pass.</li>
    <li><code>notebooks/improvement_gepa.py</code> — GEPA-style outer loop:
        N trajectories on T1 → meta-LLM proposes prompt mutations from the
        Pareto frontier → accept best on validation → repeat. Stub mutation
        logic runs end-to-end; replace with an Anthropic / OpenAI call onsite.</li>
</ul>
"""


def render() -> None:
    gr.HTML(_INTRO_HTML)
    gr.HTML(_ribbon_html())
    gr.HTML(_run_summary_html())
    gr.HTML(_PATH_HTML)
