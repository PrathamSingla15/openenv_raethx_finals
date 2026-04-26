---
title: TradeBench
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
pinned: false
license: mit
tags:
  - openenv
  - reinforcement-learning
  - finance
  - long-horizon
---

# TradeBench

**A long-horizon, non-stationary, reward-hack-resistant finance-trading RL environment for LLM agents.**

Meta PyTorch OpenEnv Hackathon (India 2026) finals submission — **Theme 2: (Super) Long-Horizon Planning & Instruction Following**.

> TradeBench is a single-agent equities-trading environment built so an LLM can be trained end-to-end on it: dense, Kelly-optimal log-wealth reward decomposed into seven attributable components; layered look-ahead defenses so the agent must derive strategy from the data it sees, not retrieve it from training memory; reward-hack safeguards so the score reflects real edge.

---

## Links (submission deliverables)

| | |
|---|---|
| HF Space (deployed env) | https://huggingface.co/spaces/yobro4619/tradebench |
| Code repository | https://github.com/PrathamSingla15/openenv_raethx_finals |
| Writeup / blog post | [`Blog.md`](Blog.md) (also at `https://huggingface.co/spaces/yobro4619/tradebench/blob/main/Blog.md`) |
| Training script | [`scripts/run_reflection_loop.py`](scripts/run_reflection_loop.py) — runnable Python entry point. No Colab notebook is needed; reproduction is a single `uv run python scripts/run_reflection_loop.py --iters 5 --rollout-model qwen/qwen3-32b:groq` against any TradeBench env URL. |

Other useful pointers:

- **Reflection-loop demo results (long form):** [`RESULTS.md`](RESULTS.md) — full per-iter trajectory, per-bar alpha vs B&H, action mix evolution, reward-component decomposition.
- **Plain-language explainer (PDF):** [`docs/tradebench-explainer.pdf`](docs/tradebench-explainer.pdf)
- **Architecture deep-dive:** [`docs/architecture/event-order.md`](docs/architecture/event-order.md), [`docs/sandbox/security-model.md`](docs/sandbox/security-model.md), [`docs/benchmark/metric-panel.md`](docs/benchmark/metric-panel.md)

---

## 1. Problem

Frontier LLMs fail at long-horizon, non-stationary decision-making under uncertainty. They can reason about a single trade in isolation, but break down across hundreds of sequential decisions where past actions reshape future state, correlations drift across regimes, and the model's own weights already encode the market history we want to test on. Naive backtests on familiar dates measure memorization, not skill.

TradeBench is designed so this class of agent can actually be **trained** (not just scored):

- **Dense, Kelly-optimal log-wealth reward** — theoretically the right objective for long-run geometric growth under uncertainty.
- **Multi-signal reward decomposition** — Sharpe bonus, drawdown penalty, turnover cost, concentration penalty, rules-compliance, anti-hack — so a GRPO trainer can attribute credit across components instead of chasing a single scalar.
- **Layered look-ahead prevention** — SQL-level PIT filter + tool-level date gating + progressive filesystem materialization + rules-based-strategy clause + post-cutoff test tier — so the agent must derive strategy, not retrieve memory.
- **Reward-hacking safeguards** — forbidden-global scanner, bounded secondary rewards, bounded turnover, periodic generation inspection.

---

## 2. Environment

### 2.1 The five-step loop

Each trading day the agent cycles through:

1. **Observe** — call `view_universe`, `view_portfolio`, `view_time`, `view_orders`, `view_constraints`, `view_episode_metrics` to get the current snapshot.
2. **Model** — call `sandbox_exec` to run Python (pandas / NumPy / scikit-learn / statsmodels) in a hardened Docker container. Read-only access to Parquet catalog (PIT-filtered). No internet.
3. **Size** — call `record_decision` with regime label, edge summary, intended exposure, top convictions, uncertainty level, and an optional `reasoning` string. `reasoning + edge_summary` are scanned by the rules-clause verifier; phrases that recall specific historical outcomes trip `r_rules = -1.0`.
4. **Place orders** — call `place_order(client_order_id, asset_id, side, quantity)`. Orders queue; no same-bar execution.
5. **Advance** — call `advance_day`. Orders fill at next-open with slippage + fees. Portfolio marked to close. Composite reward emitted. Next day's data becomes available.

### 2.2 Three task tiers

| Tier | Horizon | Universe | Manifest window | Tests |
|---|---|---|---|---|
| **T1 — Easy** | 60 bars (~3 mo) | 5 aliased assets | 2020-01-02 to 2020-03-31 | Basic Kelly sizing, short horizon |
| **T2 — Medium** | 120 bars (~6 mo) | 10 aliased assets | 2020-06-01 to 2020-11-30 | Regime navigation, recovery from drawdowns |
| **T3 — Hard** | 252 bars (~1 yr) | 20 aliased assets | 2024-01-02 to 2024-12-31 | Full season, long-horizon memory |

Each tier has a **deterministic grader** producing: final cumulative log-wealth, max drawdown, Sharpe, Sortino, avoided-ruin boolean. Every metric is computed from the ledger event log — fully reproducible, no LLM judge.

> **Real-derived, aliased, date-shifted data.** Tier bars come from real OHLCV (mega-cap US tickers via yfinance) baked once by `scripts/build_real_dataset.py`. Four anti-memorization layers stack on top: (1) every asset is exposed only as `tier_t{1,2,3}_a{NN}` aliases — real ticker symbols never appear in any tool output or sandbox file; (2) each tier picks an independent random source-window from the broad pool `[2018, today]`, so different tiers never share a regime; (3) within a tier, the alias-to-ticker mapping is permuted per build, so even recognizing "this looks like 2022-Q3" doesn't tell the model which alias is AAPL; (4) sigma=0.0005 zero-mean Gaussian return noise prevents exact-price recall. Series are rescaled to base $75 so equal-weight sizing stays well-conditioned. The (window, ticker, permutation, noise-seed) tuple is committed privately to `datasets/real-data/aliases.json` for replay-determinism audit; `progressive_fs` walks the catalog dir and never copies it into the agent sandbox.

### 2.3 Tool surface

| Tool | Purpose | Advances time? |
|---|---|---|
| `view_universe` | Tradable assets today | No |
| `view_time` | Current date, next session | No |
| `view_portfolio` | Cash, positions, marks, value | No |
| `view_orders` | Queued + open ledger orders | No |
| `view_constraints` | Long-only, leverage caps, sizing rules | No |
| `view_episode_metrics` | Running score, drawdown, Sharpe | No |
| `record_decision` | Log regime / edge / conviction / uncertainty | No |
| `place_order` | Queue a buy/sell | No |
| `cancel_order` | Cancel a queued order | No |
| `advance_day` | Execute queued orders, settle, reveal data | **Yes** |
| `sandbox_exec` | Python/shell inside isolated Docker | No |

### 2.4 Reward (composite, 7-component)

The primary signal is dense log-wealth; six bounded regularizers shape behavior without dominating.

```
r_total = r_wealth + 0.5 * (r_sharpe_bonus + r_drawdown + r_turnover
                            + r_concentration + r_rules + r_hack)

where:
  r_wealth          = log(V_{t+1} / V_t)            (unbounded, Kelly-optimal)
  r_sharpe_bonus    in [0, 0.2]                     (rolling 20-bar annualized Sharpe / 2)
  r_drawdown        in [-0.3, 0]                    (-(new-HWM dd %)^2)
  r_turnover        in [-0.1, 0]                    (-(turnover - 0.5)^+ * 0.2)
  r_concentration   in [-0.1, 0]                    (-(HHI - 0.5)^+ * 0.2)
  r_rules           in {-1.0, 0}                    (rules-based-strategy clause violation)
  r_hack            in {-1.0, 0}                    (forbidden-global detected in sandbox output)
```

Every component is surfaced in `TradeObservation.reward_breakdown` so training loops can log and plot per-signal curves. See `src/tradebench/rewards/composite.py`.

### 2.5 Look-ahead prevention — layered defense

LLM weights post-date the backtest window, so a single defense is never enough. TradeBench addresses this with six independent layers:

1. **SQL-level PIT filter** — every DuckDB query filters `available_at <= current_date`.
2. **Tool-level date gating** — every data-view tool validates its `as_of_date` argument; requests past `current_date` are rejected.
3. **Progressive filesystem view** — only files with `available_at <= current_date` are materialized into the sandbox read-only mount. Copy-on-reset + refresh-on-advance.
4. **Rules-based-strategy clause** — system prompt forbids memorized recalls ("I know X happened"). Enforced by regex scanner over agent reasoning.
5. **No internet in sandbox** — network-disabled Docker, seccomp profile, non-root.
6. **Aliased + date-shifted data** — agents see opaque `tier_t1_a01` IDs against shifted manifest dates; the underlying real series is never identifiable.

### 2.6 Reward-hacking safeguards

- **Forbidden-global scanner** — regex on every `sandbox_exec` stdout/stderr and files the agent writes. Catches `eval`, `exec`, `__import__`, `ctypes`, `subprocess`, direct ledger attribute mutation. Violation triggers `r_hack = -1.0` and episode termination.
- **Bounded secondary rewards** — no single regularizer can exceed ~0.2 per bar, so gaming any one doesn't dominate the primary wealth signal.
- **Per-tier turnover caps** — T1: 10 trades/day, T2: 25, T3: 50. Prevents penny-order exploits.
- **Periodic generation inspection** — training loop samples 1-in-N rollouts for visual review.

---

## 3. Results — Reflection-loop demo (5 iterations)

We trained the agent prompt via a **GEPA-style reflection loop**: each iteration, Claude Opus 4.7 inspects the prior rollout's trajectory, identifies failure modes, and proposes a surgical edit to the system prompt (length capped at 1.10× the prior version). Five iterations of Qwen3-32B (Groq) lifted `score_normalized` **0.6155 → 0.6584** (+4.3 pp monotonic) and ROI **+2.78% → +9.89%** (3.6× absolute improvement) on the held-out 119-bar test episode. A parallel run on GLM-5.1 (Together) is still in flight; results will be appended here and in [`RESULTS.md`](RESULTS.md) when complete.

**Reward evolution across iterations**

![Reward evolution](docs/figures/reward_evolution.png)

**ROI vs. score_normalized per iteration**

![ROI and score per iter](docs/figures/reward_roi_combined.png)

**The killer plot — per-bar cumulative log-alpha vs equal-weight buy-and-hold**

![Bar-level alpha vs B&H](docs/figures/bar_alpha_vs_bnh.png)

This is the plot that proves reflection actually taught the agent something: the baseline (red dashed) bled alpha to B&H all the way to −13 log-points by bar 119; iter-5 (green) cut that gap to −5 log-points and held flat across the back half of the episode.

| Iter | Score | ROI | place_order | sandbox_exec |
|---:|---:|---:|---:|---:|
| baseline | 0.6155 | +2.78% | 4 | 22 |
| iter 1 | 0.6214 | +4.72% | 6 | 29 |
| iter 2 | 0.6306 | +3.88% | 10 | 6 |
| iter 3 | 0.6381 | +6.31% | 11 | 1 |
| iter 4 | 0.6476 | +6.96% | 14 | 6 |
| **iter 5** | **0.6584** | **+9.89%** | **12** | **10** |

See [`RESULTS.md`](RESULTS.md) for the full comparative analysis (incl. per-bar alpha-vs-B&H, action-mix evolution, reward-component decomposition, and the Qwen-vs-GLM head-to-head once GLM completes).

---

## 4. Why it matters

- **Finance is the canonical professional long-horizon task.** Every other Theme-2 candidate (codebase refactoring, logistics, research planning) has fewer public benchmarks and murkier rewards. Finance has a clean, verifiable, Kelly-optimal reward function and rich public data.
- **Training-ready, not eval-only.** Most financial LLM benchmarks score agents post-hoc. TradeBench ships a dense composite reward designed specifically for GRPO / RLVR training — this is the env researchers will actually train on.
- **Reward hacking is the real failure mode.** Most finance envs either produce so-sparse rewards that RL can't learn, or so-exploitable rewards that the agent games them. TradeBench's multi-component design + hacker-scanner pins down both.
- **Anti-memorization is structural, not aspirational.** Aliased universe + per-tier random source windows + per-build alias-ticker permutation + return noise + post-cutoff option means the agent has to derive strategy from the data it sees, not retrieve it from weight memory.

---

## 5. Architecture

```
+-----------------------------------------------------------------+
|              OpenEnv Client / Training loop                     |
|         TRL GRPO + Unsloth via inference.py                     |
+-----------------------------------------------------------------+
|             server/app.py (FastAPI via create_app)              |
|     /reset  /step  /state  /schema  /health  /docs  /ws         |
+-----------------------------------------------------------------+
|              server/environment.py (Environment)                |
|  dispatches TradeAction -> TradeBenchSessionRuntime tool calls  |
|  composes TradeObservation (tool_output + reward_breakdown)     |
+-----------------------------------------------------------------+
|           src/tradebench/environment/session.py                 |
|   per-episode session runtime: tool dispatch, ledger            |
|   projection, advance_day, sandbox_exec                         |
+-----------------------------------------------------------------+
|  rewards/          environment/             episodes/           |
|  composite.py      date_gate.py             tiers.py            |
|  anti_hack.py      progressive_fs.py        manifests/*.json    |
+-----------------------------------------------------------------+
|   execution/engine.py    ledger/projector.py    data/query.py   |
|      (fills, fees)       (event -> portfolio)   (PIT DuckDB)    |
+-----------------------------------------------------------------+
|                  sandbox/providers/docker.py                    |
|        seccomp - no-network - read-only root - non-root         |
+-----------------------------------------------------------------+
|                       Parquet catalog                           |
|   daily_bars - fundamentals_pti - corporate_actions - calendar  |
+-----------------------------------------------------------------+
```

**Design principle:** the environment owns all portfolio state. The agent cannot directly mutate cash, positions, or scores — only inspect via tools and submit orders. The ledger event log is the single source of truth.

---

## 6. Quickstart

### Run via Docker (mirrors the HF Spaces deployment exactly)

```bash
docker build -t tradebench:latest .
docker run -p 8000:8000 tradebench:latest

curl http://localhost:8000/health
# -> {"status":"healthy"}

# Open the UI
open http://localhost:8000/web
```

### Local development (Python)

```bash
git clone <this repo> && cd trade-bench
uv sync --extra openai

# Run the OpenEnv server locally
uv run uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
# -> http://localhost:8000/web   Gradio UI with 7 tabs
# -> http://localhost:8000/docs  OpenAPI / Swagger
```

### Run a deterministic baseline (no LLM required)

```bash
# Drives the env through cash-only / equal-weight / half-Kelly across all 3 tiers
uv run python inference.py --baseline cash
uv run python inference.py --baseline equal_weight
uv run python inference.py --baseline kelly
# Same [START]/[STEP]/[END] log shape as the LLM driver - auto-validator parses both.
```

### Run the LLM driver (OpenAI-compatible)

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export OPENAI_API_KEY="..."
uv run python inference.py
# Emits [START]/[STEP]/[END] logs for t1/t2/t3
```

### Run the verifier (assert env claims)

```bash
uv run python -m tradebench.verifier --tier t1 --seed 42
# RESULT: PASS  ->  conformance, leak, replay, determinism all green
```

### Drive the env from a Python client

```python
import asyncio
from tradebench import TradeBenchEnv, TradeAction

async def main():
    async with TradeBenchEnv(base_url="https://yobro4619-tradebench.hf.space") as env:
        result = await env.reset(task_tier="t1")
        result = await env.step(TradeAction(action_type="view_portfolio"))
        result = await env.step(TradeAction(
            action_type="place_order",
            payload={
                "client_order_id": "o1",
                "asset_id": "tb_sample_liquid",
                "side": "buy",
                "quantity": 10,
            },
        ))
        result = await env.step(TradeAction(action_type="advance_day"))
        print(result.observation.reward_breakdown)

asyncio.run(main())
```

---

## 7. Repo layout

```
openenv_raethx_finals/
|-- openenv.yaml                  # OpenEnv manifest (spec_version: 1)
|-- pyproject.toml                # Hatchling build, openenv-core dep
|-- uv.lock                       # Pinned dependency graph
|-- models.py                     # TradeAction / TradeObservation / TradeState
|-- client.py                     # TradeBenchEnv (EnvClient subclass)
|-- inference.py                  # LLM driver (Qwen3-32B / GLM-5.1 via HF Inference)
|-- reflection.py                 # GEPA-style Opus 4.7 reflector
|-- inference_logger.py           # Trajectory + summary writer
|-- __init__.py                   # Top-level re-exports
|-- server/
|   |-- app.py                    # FastAPI create_app + Gradio mount
|   |-- tradebench_environment.py # Environment subclass — wraps session runtime
|   |-- Dockerfile                # Canonical OpenEnv multi-stage build
|   `-- ui/                       # Multi-tab Gradio web interface
|-- src/tradebench/               # Internal library
|   |-- rewards/                  # composite.py (7-component [0,1] convex × gate) - anti_hack.py
|   |-- environment/              # session - date_gate - progressive_fs - prompt - metrics - tools
|   |-- episodes/                 # tiers - loader - models - study_packet
|   |-- data/                     # PIT DuckDB query - catalog - sample / real dataset builders
|   |-- execution/                # Order engine, slippage, corporate actions
|   |-- ledger/                   # Event log + projector
|   |-- sandbox/                  # Docker / local provider with seccomp + no-net
|   |-- scoring/                  # Episode metrics
|   |-- baselines/                # cash - equal_weight - kelly
|   `-- verifier/                 # leak - replay - determinism - conformance - cli
|-- docker/                       # Sandbox Dockerfile + seccomp profile + constraints
|-- datasets/                     # real-data/ and catalog/ — committed for replay
|-- artifacts/                    # Reflection-loop trajectories + reflector artifacts (proof)
|   |-- runs/                     # 8 rollouts: Qwen baseline + 5 reflection iters; GLM baseline + iter1
|   `-- reflection_*/             # Reflector meta-prompts, generated system prompts, history.json
|-- scripts/
|   |-- run_reflection_loop.py    # End-to-end GEPA loop (--iters N)
|   |-- replay_baseline_reward.py # Recompute new convex composite over a recorded run
|   |-- profile_rollout.py        # Wall-time / token profiling
|   |-- build_real_dataset.py     # One-shot real-data → aliased catalog builder
|   `-- visualize/                # 6 figure-generation scripts + render_all.py
|-- docs/
|   |-- tradebench-explainer.{md,pdf}  # 7-page plain-language explainer
|   |-- architecture/, sandbox/, benchmark/, data/  # Deep-dive markdown
|   `-- figures/                  # All 6 PNGs embedded in this README + RESULTS.md
|-- tests/                        # pytest — data, execution, ledger, sandbox, environment, rewards, verifier
|-- README.md                     # ← you are here
`-- RESULTS.md                    # Full reflection-demo narrative + comparative analysis
```

---

## 8. What's distinctive about TradeBench

| Capability | TradeBench |
|---|---|
| Domain | US equities, 60 / 120 / 252-bar tiers |
| Reward | Dense log-wealth + 6 bounded regularizers, surfaced as a 7-component breakdown |
| Walk-forward window | Real-derived OHLCV, aliased and date-shifted |
| Tool-level date gate | `as_of_date` validated on every data tool |
| Progressive FS | Copy-on-reset, refresh-on-advance |
| Rules-based clause | Regex scanner over `reasoning` + `edge_summary` |
| Forbidden-global scanner | 12 patterns, scanned on every `sandbox_exec` |
| Reward-hack bounds | sharpe in [0, 0.2]; dd, turnover, conc in [-0.1, 0]; rules/hack in {-1, 0} |
| Multi-task tiers | 3 tiers (60 / 120 / 252 bars) |
| Verifier | conformance + leak + replay + determinism + termination + edge |
| Reference baselines | cash / equal-weight / Kelly via `inference.py --baseline` |
| OpenEnv-compliant | Yes |

---

## 9. Open issues and known limitations

- **Real OHLCV under aliases (memorization-defended).** T1/T2/T3 bars are real mega-cap data baked via `scripts/build_real_dataset.py`. Anonymization is the four-layer stack described in section 2.2 — alias renaming, per-tier random source windows, alias-to-ticker permutation per build, plus sigma=0.0005 return noise. Curating a survivorship-corrected universe (today's pool is hand-picked gap-free names) is out of scope for the hackathon.
- **Rules-clause scanner is regex-only.** It catches first-person memorized recalls (`"I recall AAPL crashed in 2020"`) but a paraphrased recall (`"the period had a sharp downturn"`) can slip through. Strong as a training signal, not a proof of no memorization. The aliased + date-shifted data is the structural safeguard.
- **Progressive FS is copy-based, not FUSE.** Works but uses extra disk per episode. The current `progressive_fs._should_copy` falls open on flat-parquet files (sample dataset) and gates strictly on date-partitioned paths.
- **Single-agent design.** Multi-agent structured-report architectures are a stretch goal, not shipped — this submission is single-agent (Theme 2 framing).
- **Training evidence** is generated onsite; baseline-vs-trained plots are committed to `docs/plots/` after the run.

---

## 10. License

MIT. See the Meta OpenEnv Hackathon terms for submission-specific provisions.

---

## Acknowledgements

The Meta PyTorch OpenEnv team for the framework and reference environments. Scaler School of Technology for organising the hackathon.
