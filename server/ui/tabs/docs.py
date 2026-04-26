"""Docs tab: deployment links, repo file table, citation."""

from __future__ import annotations

import gradio as gr


_LINKS_HTML = """
<div class="tb-section-eyebrow">deployment &middot; code &middot; companion docs</div>
<h2 class="tb-section-title">
    <span class="tb-faded">Everything a reviewer needs to inspect the env,</span>
    re-run it, or cite it.
</h2>

<div class="tb-link-grid">
    <a class="tb-link-card" href="https://huggingface.co/spaces/yobro4619/tradebench" target="_blank">
        <div class="tb-link-eyebrow">huggingface space</div>
        <div class="tb-link-title">yobro4619 / tradebench</div>
        <div class="tb-link-sub">live env · this UI · OpenEnv-compliant</div>
    </a>
    <a class="tb-link-card" href="https://github.com/PrathamSingla15/openenv_raethx_finals" target="_blank">
        <div class="tb-link-eyebrow">github</div>
        <div class="tb-link-title">PrathamSingla15 / openenv_raethx_finals</div>
        <div class="tb-link-sub">source · 322 files · MIT license</div>
    </a>
    <a class="tb-link-card" href="https://yobro4619-tradebench.hf.space/docs" target="_blank">
        <div class="tb-link-eyebrow">openapi · /docs</div>
        <div class="tb-link-title">interactive api</div>
        <div class="tb-link-sub">/reset · /step · /state · /schema</div>
    </a>
    <a class="tb-link-card" href="https://yobro4619-tradebench.hf.space/health" target="_blank">
        <div class="tb-link-eyebrow">health probe</div>
        <div class="tb-link-title">/health</div>
        <div class="tb-link-sub">cold-start check</div>
    </a>
    <a class="tb-link-card" href="https://github.com/PrathamSingla15/openenv_raethx_finals/blob/main/Blog.md" target="_blank">
        <div class="tb-link-eyebrow">writeup</div>
        <div class="tb-link-title">Blog.md</div>
        <div class="tb-link-sub">long-form · why reflection beats GRPO here</div>
    </a>
    <a class="tb-link-card" href="https://github.com/PrathamSingla15/openenv_raethx_finals/blob/main/RESULTS.md" target="_blank">
        <div class="tb-link-eyebrow">long-form results</div>
        <div class="tb-link-title">RESULTS.md</div>
        <div class="tb-link-sub">per-iter trajectory · component decomposition</div>
    </a>
    <a class="tb-link-card" href="https://colab.research.google.com/github/PrathamSingla15/openenv_raethx_finals/blob/main/notebooks/training.ipynb" target="_blank">
        <div class="tb-link-eyebrow">colab · open in browser</div>
        <div class="tb-link-title">notebooks/training.ipynb</div>
        <div class="tb-link-sub">runs the full reflection loop end-to-end</div>
    </a>
</div>
"""


_REPO_HTML = """
<div class="tb-section-eyebrow">repository &middot; what's where</div>
<h2 class="tb-section-title">
    <span class="tb-faded">The map of the repo</span>
    that produced everything above.
</h2>

<table class="tb-table">
    <thead>
        <tr><th>Path</th><th>Role</th></tr>
    </thead>
    <tbody>
        <tr><td class="mono">README.md</td><td>HF Space frontmatter + project overview + quickstart + limitations.</td></tr>
        <tr><td class="mono">Blog.md · RESULTS.md</td><td>Long-form writeup (self-contained) + reflection-loop demo trajectory.</td></tr>
        <tr><td class="mono">models.py · client.py · openenv.yaml</td><td>OpenEnv wire contracts and manifest at repo root.</td></tr>
        <tr><td class="mono">Dockerfile · server/app.py</td><td>HF-Space root Dockerfile + FastAPI app with Gradio at <code>/</code>.</td></tr>
        <tr><td class="mono">server/tradebench_environment.py</td><td>OpenEnv Environment subclass; reset / step / state.</td></tr>
        <tr><td class="mono">src/tradebench/rewards/composite.py</td><td>7-component RewardBreakdown + <code>compute_composite_reward</code>.</td></tr>
        <tr><td class="mono">src/tradebench/rewards/anti_hack.py</td><td><code>scan_forbidden_globals</code> + <code>check_rules_clause</code>.</td></tr>
        <tr><td class="mono">src/tradebench/environment/{date_gate, progressive_fs, session, prompt}.py</td><td>Look-ahead defenses + the runtime they wire into.</td></tr>
        <tr><td class="mono">src/tradebench/episodes/tiers.py</td><td>train / test / t1 manifest lookup.</td></tr>
        <tr><td class="mono">src/tradebench/baselines/{cash, equal_weight, kelly}.py</td><td>Three async reference strategies.</td></tr>
        <tr><td class="mono">src/tradebench/verifier/</td><td>conformance · leak · replay · determinism · CLI.</td></tr>
        <tr><td class="mono">scripts/run_reflection_loop.py</td><td>End-to-end reflection loop driver.</td></tr>
        <tr><td class="mono">scripts/visualize/</td><td>5 figure-generation scripts + <code>render_all.py</code>.</td></tr>
        <tr><td class="mono">notebooks/training.ipynb</td><td>Colab-runnable reflection loop notebook.</td></tr>
        <tr><td class="mono">artifacts/</td><td>Reflection-loop trajectories + reflector artifacts (Qwen 5 iters, GLM 3 iters).</td></tr>
        <tr><td class="mono">datasets/</td><td>Aliased real OHLCV catalog (sample-v2 + real-v0).</td></tr>
        <tr><td class="mono">docs/figures/</td><td>11 PNGs embedded in README + RESULTS.md + Blog.md.</td></tr>
    </tbody>
</table>
"""


_CITATION_HTML = """
<div class="tb-section-eyebrow">citation</div>
<h2 class="tb-section-title">
    <span class="tb-faded">If TradeBench is useful in your work,</span>
    cite as:
</h2>

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
Scaler School of Technology for organising the hackathon. Anthropic for the
Opus 4.7 reflector via OpenRouter, and the HuggingFace Inference Providers
team for the Groq + Together model serving that made 5-iter reflection in
a hackathon weekend possible at all.
</p>
"""


def render() -> None:
    gr.HTML(_LINKS_HTML)
    gr.HTML(_REPO_HTML)
    gr.HTML(_CITATION_HTML)
    gr.HTML(_ACK_HTML)
