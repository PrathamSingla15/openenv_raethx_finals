"""Docs & Links tab — pointers + citation block."""

from __future__ import annotations

import gradio as gr


_LINKS_HTML = """
<div class="tb-section-eyebrow">docs · code · model · datasets</div>
<h2 class="tb-section-title">Everything a teammate or a judge needs to
inspect the env, re-run it, or cite it.</h2>

<div class="tb-metric-row">
    <a href="https://huggingface.co/spaces/yobro4619/tradebench" target="_blank"
       style="text-decoration: none; color: inherit;">
        <div class="tb-metric">
            <div class="tb-metric-label">huggingface space</div>
            <div class="tb-metric-value" style="font-size: 1.1em; font-family: var(--type-display); font-weight: 500;">yobro4619 / tradebench</div>
            <div class="tb-metric-sub">live env, this UI</div>
        </div>
    </a>
    <a href="https://yobro4619-tradebench.hf.space/docs" target="_blank"
       style="text-decoration: none; color: inherit;">
        <div class="tb-metric">
            <div class="tb-metric-label">openapi · /docs</div>
            <div class="tb-metric-value" style="font-size: 1.1em; font-family: var(--type-display); font-weight: 500;">interactive api</div>
            <div class="tb-metric-sub">/reset · /step · /state · /schema</div>
        </div>
    </a>
    <a href="https://yobro4619-tradebench.hf.space/health" target="_blank"
       style="text-decoration: none; color: inherit;">
        <div class="tb-metric">
            <div class="tb-metric-label">health probe</div>
            <div class="tb-metric-value" style="font-size: 1.1em; font-family: var(--type-display); font-weight: 500;">/health</div>
            <div class="tb-metric-sub">cold-start check</div>
        </div>
    </a>
</div>
"""


_REPO_HTML = """
<div class="tb-section-eyebrow">repository · what's where</div>

<table class="tb-table">
    <thead>
        <tr><th>Path</th><th>Role</th></tr>
    </thead>
    <tbody>
        <tr><td class="mono">IMPROVEMENT_PLAN.md</td><td>Canonical plan: reward, walk-forward, verifier, roadmap. Source-of-truth for the teammate PDF.</td></tr>
        <tr><td class="mono">README.md</td><td>HF Spaces frontmatter + project overview + quickstart + limitations.</td></tr>
        <tr><td class="mono">models.py · client.py · openenv.yaml</td><td>OpenEnv wire contracts and manifest at repo root.</td></tr>
        <tr><td class="mono">server/app.py · Dockerfile</td><td>FastAPI + Gradio mount; multi-stage Docker matching HF Spaces.</td></tr>
        <tr><td class="mono">server/tradebench_environment.py</td><td>OpenEnv Environment subclass; reset / step / state.</td></tr>
        <tr><td class="mono">src/tradebench/rewards/composite.py</td><td>7-component RewardBreakdown + compute_composite_reward.</td></tr>
        <tr><td class="mono">src/tradebench/rewards/anti_hack.py</td><td>scan_forbidden_globals + check_rules_clause.</td></tr>
        <tr><td class="mono">src/tradebench/environment/{date_gate,progressive_fs,session,prompt}.py</td><td>Look-ahead defenses + the runtime they wire into.</td></tr>
        <tr><td class="mono">src/tradebench/episodes/tiers.py</td><td>T1 / T2 / T3 manifest lookup by task_id.</td></tr>
        <tr><td class="mono">src/tradebench/baselines/{cash,equal_weight,kelly}.py</td><td>Three async reference strategies.</td></tr>
        <tr><td class="mono">src/tradebench/verifier/</td><td>conformance · leak · replay · determinism · CLI.</td></tr>
        <tr><td class="mono">notebooks/improvement_gepa.py · training_grpo.py</td><td>Path B headline + Path A RL-readiness proof.</td></tr>
        <tr><td class="mono">scripts/generate_plots.py · capture_screenshots.py</td><td>Plot generator + Playwright screenshot harness.</td></tr>
    </tbody>
</table>
"""


_CITATION_HTML = """
<div class="tb-section-eyebrow">citation</div>
<h2 class="tb-section-title">If TradeBench is useful in your work, cite as:</h2>

<details class="tb-schema" open>
    <summary>BibTeX</summary>
<pre>@misc{tradebench2026,
  title  = {{TradeBench}: A Long-Horizon, Reward-Hack-Resistant Trading
            RL Environment for {LLM} Agents},
  author = {TradeBench Contributors},
  year   = {2026},
  note   = {Meta PyTorch OpenEnv Hackathon (India 2026), Theme 2.
            \\url{https://huggingface.co/spaces/yobro4619/tradebench}},
}</pre>
</details>
"""


_ACK_HTML = """
<div class="tb-section-eyebrow">acknowledgements</div>
<p class="tb-section-lead tb-section-lead-mute">
The Meta PyTorch OpenEnv team for the framework and reference environments.
Scaler School of Technology for organising the hackathon.
</p>
"""


def render() -> None:
    gr.HTML(_LINKS_HTML)
    gr.HTML(_REPO_HTML)
    gr.HTML(_CITATION_HTML)
    gr.HTML(_ACK_HTML)
