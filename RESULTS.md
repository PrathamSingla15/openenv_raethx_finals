# TradeBench — Reflection-loop demo results

This document is the long-form companion to the [README "Results" section](README.md#3-results--reflection-loop-demo-5-iterations).

It walks through:

1. The setup we ran
2. **Qwen3-32B (Groq)** — full 5-iter trajectory, with the Opus diagnoses that drove each prompt edit
3. Per-bar performance vs equal-weight buy-and-hold (the most diagnostic plot)
4. What the agent learned to do differently — action mix and reward-component evolution
5. **GLM-5.1 (Together)** — parallel run, currently in flight
6. Comparative analysis (Qwen vs GLM) — gated until GLM completes
7. The reflection mechanism — prompt evolution
8. Reproducibility — exact command lines

---

## 1. Setup

Two open-weights models, identical reflection harness, identical evaluation:

| Setting | Value |
| --- | --- |
| Test episode | `tier_test`, 119 bars, 2024-12-19 → 2025-06-04 |
| Universe | 10 aliased equities (`tier_a01` … `tier_a10`), real OHLCV underneath |
| Initial capital | $100,000 |
| Reward function | 7-component convex composite × compliance gate, all in [0, 1] |
| Reflector | Anthropic Claude Opus 4.7 via OpenRouter, 1.10× length-budget cap on prompt edits |
| Iterations | 5 |
| Per-bar gate | `record_decision` mandatory before every `advance_day` (server-enforced) |
| Score reported | `score_normalized` = mean per-bar reward over 119 bars (∈ [0, 1] by construction) |

**Baseline agent prompt** is the canonical TradeBench system prompt with the 5-step daily protocol (Observe → Model → Size → Place orders → Advance), the rules-clause anti-memorization paragraph, and a worked example. Each reflection iteration produces a *surgical edit* of the prior prompt — no rewrites.

---

## 2. Qwen3-32B (Groq) — 5 reflection iterations

![Reward evolution](docs/figures/reward_evolution.png)

![ROI and score per iter](docs/figures/reward_roi_combined.png)

### Per-iter trajectory

| Iter | Score | ROI | place_order | sandbox_exec | view_* |
|---:|---:|---:|---:|---:|---:|
| 0 — baseline | 0.6155 | +2.78% | 4 | 22 | 6 |
| 1 — reflection | 0.6214 | +4.72% | 6 | 29 | 23 |
| 2 — reflection | 0.6306 | +3.88% | 10 | 6 | 17 |
| 3 — reflection | 0.6381 | +6.31% | 11 | 1 | 32 |
| 4 — reflection | 0.6476 | +6.96% | 14 | 6 | 32 |
| **5 — reflection** | **0.6584** | **+9.89%** | **12** | **10** | **31** |

Score moved **monotonically** from baseline through iter 5 (+4.3 pp on the [0,1] scale), and ROI grew **3.6×** in absolute terms (+2.78% → +9.89%). Place-order count tripled. Sandbox usage spiked early (Opus told the agent to actually compute) then settled — once the agent built its mental model in iters 1–2, it converged on a more decisive policy.

### Iter 1 — Opus diagnosis

> *"After bar 1 the agent placed exactly ONE order (60 shares tier_a01) and then never placed another order for the remaining 119 bars despite recording 'diversified_long' / 'momentum_long' decisions naming 3-5 convictions each bar — the record_decision step has become a substitute for trading, leaving the portfolio severely under-exposed (60% single-name instead of the committed 80% 5-asset sharpe-weighted basket)."*

**Surgical edit:** §4.2 record_decision clarification ("logged top_convictions are intent only; subsequent place_order calls must translate them into position gaps, not be skipped"); §5 step-4 from "ZERO OR MORE times" to a conditional trigger on conviction-vs-current-weight gap.

### Iter 5 — Opus diagnosis

> *"After the initial build on bar 1 the agent essentially never re-checks the gap between its recorded 0.60 intended_exposure and its actual ~0.20 gross — it records new decisions and advances for 100+ bars without ever firing the remaining build legs, yielding only 14 place_orders over 119 bars."*

**Surgical edit:** §5 step-4 strengthened to also cover bars 2–3 when the initial build is incomplete; §7.5 named the exact failure pattern ("two legs filled out of five committed") so the agent could recognize itself doing it.

This pattern — Opus naming the *specific behavioral failure* and the prompt absorbing a one-paragraph patch — repeated across all 5 iterations.

---

## 3. Per-bar performance vs equal-weight buy-and-hold

![Bar-level alpha vs B&H](docs/figures/bar_alpha_vs_bnh.png)

The single most informative plot in this report. Each line is the agent's **cumulative log-alpha relative to a do-nothing equal-weight buy-and-hold portfolio** of the same 10 assets. Negative = agent is behind B&H; positive = agent is beating B&H.

- **Baseline (red dashed)**: bled alpha consistently from bar ~25 onward. By bar 65 the agent was −13 log-points below B&H, and by bar 119 it was still −12 points down. The agent took risk early then stopped trading; B&H grew while the agent's static, under-exposed book lagged.
- **Final iter 5 (green solid)**: tracked B&H more closely through bars 0–40, troughed at −6 around bar 65, and *recovered* in the back half of the episode — cutting the alpha gap roughly in half by bar 119 (−5 vs −12).

The reflection loop did not turn the agent into an alpha-generating machine on this data slice (it still loses to B&H), but it visibly closed the gap in a way that's only possible if the agent is actually executing on its recorded convictions rather than freezing.

---

## 4. What the agent learned to do differently

![Action mix per iter](docs/figures/action_mix_evolution.png)

Two clear patterns:

- **place_order: 4 → 12 (3.0× growth).** This is the headline behavioral change. Every reflection iteration after iter 0 increased order count except the last (iter 5 dipped from 14 → 12 — the agent learned to be selective, not just busy).
- **sandbox_exec: 22 → 29 → 6 → 1 → 6 → 10.** The agent front-loaded computation early ("understand the universe"), then collapsed sandbox usage as it learned to trust its initial mental model and act decisively. By iter 5 it was running ~10 sandbox calls per episode — enough to validate alpha hypotheses, not so many that it analysis-paralyzed.
- **view_* (any state-inspection view): 6 → 31.** Big jump from iter 0 to iter 1 and stable thereafter — Opus told the agent to actually look at portfolio state before deciding, and the agent obeyed.

![Reward component decomposition](docs/figures/reward_components_baseline_vs_final.png)

Replaying the trajectories through the live convex composite reward function and averaging per-bar contributions:

| Component | Baseline | Final | Δ |
| --- | ---: | ---: | ---: |
| `c_alpha` (vs equal-weight B&H) | 0.401 | 0.434 | **+0.033** |
| `c_return` (cumulative log-return) | 0.508 | 0.531 | **+0.023** |
| `c_drawdown` | 0.995 | 0.964 | −0.031 |
| `c_solvency` | 0.998 | 0.998 | ±0.000 |
| `c_efficiency` (turnover) | 1.000 | 1.000 | ±0.000 |
| `c_diversity` (HHI) | 0.889 | 0.889 | ±0.000 |
| `c_consistency` (downside semi-vol of bar-alpha) | 0.665 | 0.700 | **+0.035** |

The agent improved `c_alpha`, `c_return`, and `c_consistency` in roughly equal measure — exactly the components that move *only* when the policy actually trades into its convictions. It paid for that with a small `c_drawdown` cost (more positions = more interim mark-to-market noise), which is the right trade. The flat components (solvency, efficiency, diversity) were already at or near 1.0 — there was no headroom to move them.

> Note: `c_efficiency`, `c_diversity`, and `c_drawdown` use heuristic per-bar inputs in the replay (turnover ratio = 0.05, HHI = 0.20) since the v1 trajectory.jsonl format doesn't capture those directly. The deltas are within expected noise for those columns; the alpha/return/consistency deltas are exact.

---

## 5. GLM-5.1 (Together) — 5 iterations (in progress)

The same reflection harness is currently running on `zai-org/GLM-5.1` via Together AI. As of writing:

| Iter | Status | Score | ROI |
|---:|---|---:|---:|
| 0 — baseline | complete | 0.6243 | +2.83% |
| 1 — reflection | **in flight** | TBD | TBD |

GLM's baseline already exhibits a very different policy pattern from Qwen's:

- **place_order = 2** (vs Qwen's 4) — even more reluctant to trade.
- **sandbox_exec = 81** (vs Qwen's 22) — heavy "thinking" in the sandbox before acting.
- **view_* = 88** (vs Qwen's 6) — heavy state observation.

So GLM has a *think-first-act-rarely* pattern; Qwen has a *act-quickly-think-rarely* pattern. Whether reflection drives both toward the same converged behavior, or they end up in different attractors, will be the most interesting comparison once GLM finishes.

This section will be updated when the GLM 5-iter run completes.

---

## 6. Comparative analysis — Qwen vs GLM

> Gated on GLM completion. To be filled with: best score per model, place_order count delta, sandbox_exec usage delta, biggest single-iter jump per model, prompt-evolution diff overlap, per-bar alpha-vs-B&H curves overlaid.

---

## 7. Reflection mechanism

![System-prompt evolution](docs/figures/prompt_evolution.png)

Top card is the **baseline system prompt** (399 lines after wrap). Bottom is the **iter-5 prompt** (445 lines). Red regions in the top card are sections that were removed or rewritten across the 5 reflections; green regions in the bottom card are the additions.

The diff is concentrated in two areas:

- **§5 PER-BAR PROTOCOL step 4** — initially "place_order ZERO OR MORE times". After 5 reflections, this became "If ANY named conviction still has current weight materially below its target (e.g. gap > ~2% of portfolio), issue a place_order this bar; only skip orders when current positions already approximate the recorded top_convictions or when the regime is genuinely flat."
- **§7.5 COMMON MISTAKES** — gained explicit named patterns ("two legs filled out of five committed", "recording a decision naming convictions but never sizing them is a form of standing still") so the agent could pattern-match its own behavior against known failure modes.

The compliance clause, anti-memorization rules, and worked example were left almost untouched across all 5 reflections — Opus correctly identified that those parts were already doing their job and didn't waste its 1.10× length budget on them.

---

## 8. Reproducibility

```bash
# Bring up the env locally (or use the deployed HF Space directly)
uv sync --extra openai
uv run uvicorn server.app:app --host 0.0.0.0 --port 8000 &

# 5-iter reflection run
ENV_URL="http://127.0.0.1:8000" \
HF_TOKEN="<your hf token — needed for the Inference Provider>" \
OPENROUTER_API_KEY="<your openrouter key — for the Opus reflector>" \
MODEL_NAME="qwen/qwen3-32b:groq" \
TRADEBENCH_DATASET_ROOT=$(pwd)/datasets \
uv run python scripts/run_reflection_loop.py --iters 5 \
  --rollout-model "qwen/qwen3-32b:groq"

# Continue from a prior run instead of starting fresh
uv run python scripts/run_reflection_loop.py --iters 5 \
  --from-run artifacts/runs/20260426T014336Z__train-test__qwen-qwen3-32b-groq \
  --rollout-model "qwen/qwen3-32b:groq"

# Regenerate every figure embedded in this document
uv run python -m scripts.visualize.render_all
```

Reflector model is set in `reflection.py` (`DEFAULT_REFLECTION_MODEL = "anthropic/claude-opus-4.7"`); override with `--reflection-model`. Each iteration writes:

- `artifacts/reflection_<ts>/iter_NN__reflection/meta_prompt.txt` — the prompt sent to the reflector
- `artifacts/reflection_<ts>/iter_NN__reflection/reflector_raw_response.txt` — Opus's full STEP 0/1/2 + new prompt
- `artifacts/reflection_<ts>/iter_NN__reflection/new_system_prompt.txt` — the parsed system prompt fed to the next rollout
- `artifacts/runs/<ts>__...__iterNN_reflect/{train,test}/{trajectory.jsonl,summary.json,system_prompt.txt}` — the rollout itself

All artifacts referenced in this document live under `artifacts/` in this repo — pull them, replay them, audit them.
