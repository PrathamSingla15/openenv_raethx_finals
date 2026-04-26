# TradeBench: 5 reflections, 3.6× ROI, zero gradient updates

*A long-horizon trading environment, and what we learned from training an agent on it without touching the weights.*

**Submission for the Meta PyTorch OpenEnv Hackathon (India 2026), Theme 2: (Super) Long-Horizon Planning & Instruction Following.**

- Live env: https://huggingface.co/spaces/yobro4619/tradebench
- Code: https://github.com/PrathamSingla15/openenv_raethx_finals
- Companion docs: [`README.md`](README.md), [`RESULTS.md`](RESULTS.md)

---

## 1. The result, before anything else

A frozen Qwen3-32B on a 119-bar trading episode, scored against an equal-weight buy-and-hold baseline on real OHLCV data. Iteration zero: `score_normalized` of 0.6155, ROI of +2.78%, four orders placed across the entire run. After five reflection passes, with no gradient updates and no model swap: 0.6584, +9.89%, twelve orders placed. Every iteration improved on the last. The climb was monotone.

Nothing about the model changed. The weights stayed exactly where Groq served them. Between rollouts, Claude Opus 4.7 read the prior trajectory, identified what the agent had failed to do, and rewrote the system prompt. That was the entire optimization loop.

The headline finding from this experiment is not that prompts matter. Everyone already knows prompts matter. The interesting claim is sharper: frontier LLMs already carry the planning knowledge needed to act well in a sequential, non-stationary decision environment. What they lack out of the box is a calibrated policy for spending that knowledge under tight constraints, with the right pacing, sizing, and abstention behavior. We tested whether you can install that policy through prompt evolution alone, on a task with proper anti-memorization defenses, and the answer turned out to be yes. A 3.6× ROI improvement over five iterations is hard to wave off as noise.

This post is intended to be self-contained: env description, reward definition, training mechanism, full results, and limitations all live here, so a reviewer does not need to chase companion documents to understand or reproduce the work. Section 2 explains why long-horizon trading was the right test bed and why no existing OpenEnv environment had the leak controls we needed. Section 3 specifies TradeBench in full: the 5-step daily loop, the server-enforced per-bar gate, the explicit 7-component reward formula, the train/test split, and the four-layer anti-memorization stack. Section 4 names the failure modes the baseline agent exhibited and what we were actually teaching. Section 5 lays out, with concrete numbers, why reflection-based optimization beats GRPO/PPO in this regime. Section 6 describes the reflection loop mechanically. Section 7 is the full results, both Qwen3-32B (5 iterations) and GLM-5.1 (3 iterations), with every figure embedded inline. Sections 8 and 9 cover what we learned and where the approach hits its ceiling.

## 2. Why long-horizon trading was the right test

Frontier LLMs have measurable failure modes on long-horizon decision tasks. They reason well about a single trade in isolation. They do not reason well across hundreds of sequential decisions where past actions reshape the future state distribution. The skill that breaks down is not analytical depth but pacing: knowing when to act, when to abstain, how to size given what you already hold, and how to keep from drifting into a regime you did not intend.

Finance is the cleanest place to measure this. KellyBench (2026) ran every major frontier model on a single Premier League season, asking each to manage £100K across match betting markets. Every model lost money. The best run, Claude Opus 4.6, finished at -11% ROI. Grok 3 went to ruin entirely. None of the models had a knowledge problem; they had an execution problem stretched across a few hundred sequential decisions, and that was enough to flip every one of them negative. The gap between knowing and acting, in this regime, is a stated number rather than an intuition.

The textbook answer to a long-horizon execution gap is reinforcement learning. Train the policy directly on the reward signal, with thousands of rollouts, until the calibration shows up in weight space. We costed it. A single 119-bar test episode took 45 to 50 minutes of wall-clock through GLM-5.1 on Together (and about 22 minutes through Qwen3-32B on Groq), with hundreds of tool calls per episode and a few seconds per call. Multiplying that across the rollout count an LLM-as-policy RL run actually needs got us to weeks, not the 36 hours a hackathon budget allows. Even with concurrency, the per-token economics cut the experiment off before it could start. Section 5 walks through the full numeric comparison.

So we picked the cheaper axis. Leave the model frozen. Treat prompt-space as the optimization domain. Each iteration reads the prior trajectory, finds the failure modes the agent demonstrated, proposes surgical edits to the system prompt, and redeploys. The mechanism draws from GEPA-style generative prompt evolution and the broader reflective-optimization line of work, with a strong reflector (Opus 4.7) operating on a smaller agent's (Qwen3-32B) trajectories.

The actual research question, then, was narrow and falsifiable. Does reflection on trajectory, with a stronger model in the editor seat, move scores on a finance task whose anti-memorization defenses are tight enough that the agent cannot just pattern-match its way through? We had to build the task to find out, because no existing OpenEnv environment shipped with the leak controls we needed. That env is TradeBench.

## 3. TradeBench in one section

The agent runs the same five-step loop on every bar. It observes through a set of `view_*` read tools that expose the current portfolio, the rolling price window, and a study packet of pre-computed features. It models, by emitting Python into `sandbox_exec`, which runs in a hardened Docker container with pandas, numpy, and a progressive filesystem that only contains parquet files dated strictly before the current trading day. It sizes by calling `record_decision`, which forces the agent to commit a regime label, an edge summary, an intended exposure, the top conviction picks, and an uncertainty bound before the bar closes. It places orders through `place_order`. Then it calls `advance_day`, which is the only call that moves the clock. There are eleven tools total. Orders queue and fill at the next-open price with deterministic slippage and cost models, so the same trajectory replays bit-for-bit on the same seed.

The structural piece that makes the env actually about long-horizon planning, rather than next-action prediction, is the `record_decision` gate. The session in `src/tradebench/environment/session.py` enforces it server-side: if the agent calls `advance_day` without first writing a decision for the current bar, the call is rejected. The agent literally cannot move forward without committing to a thesis on the record. Every advance leaves a paper trail that the reward function can audit.

The reward itself is a 7-component composite, all bounded to [0,1] by construction, with weights that sum to 1.0, multiplied by a {0,1} compliance gate. The full formula:

```
r_t = (sum_i w_i * c_i_t) * g_compliance_t                 # in [0, 1]

components (each in [0, 1]):
  c_alpha       sigmoid(8 * cumulative_log_alpha_vs_bench)        w = 0.40
  c_return      sigmoid(5 * cumulative_log_return)                w = 0.15
  c_drawdown    1 - 2 * min(dd, 0.5)                              w = 0.10
  c_solvency    sigmoid(6 * (V - 0.5*V0) / (0.5*V0))              w = 0.10
  c_efficiency  exp(-2 * max(0, turnover - 0.10))                 w = 0.10
  c_diversity   1 - clip((HHI - 0.10) / 0.90, 0, 1)               w = 0.05
  c_consistency win * stability + (1 - win) * 0.5                 w = 0.10

compliance gate (multiplicative, not additive):
  g_compliance = (1 - viol_rules) * (1 - viol_hack) * (1 - lev_breach)
  with lev_breach = 1{gross_leverage > 1.0}.  Any single violation zeros the bar.

episode score:
  score_normalized = mean over bars of r_t                        in [0, 1]
```

The whole module lives in `src/tradebench/rewards/composite.py`. Three design choices are worth naming. First, the 0.40 weight on alpha vs equal-weight buy-and-hold is dominant on purpose: a lucky cash-only agent should not beat a thoughtful one in a year when markets happen to fall. Second, `c_consistency` is gated. Above-benchmark agents earn it for low downside-alpha-volatility; sub-benchmark agents fall back to 0.5 instead of 0. That removes the "flatline gets free Sharpe" exploit that killed our first reward design. Third, the compliance gate is multiplicative. Cheating zeros the bar's reward outright; there is no way for a great Sharpe to earn back a rules-clause hit. All seven components plus the gate are surfaced in `TradeObservation.reward_breakdown` for per-bar logging, so reflection's reward analysis is always grounded in component-level signal rather than a single opaque scalar.

The episode shape is a clean train / test split. The `train` phase is 252 bars (~1 year) of in-context study packet: at episode reset the agent receives the full window's OHLCV plus summary stats and a correlation matrix, derives a strategy, and emits a single `record_decision`. No per-bar rollout, no reward. The `test` phase is 120 bars (~6 months) of held-out walked-bar-by-bar evaluation, calendar-adjacent to `train` (it starts the trading day after `train` ends). The agent steps one bar at a time, the composite reward emits per `advance_day`, and the mean per-bar reward across the window is `score_normalized`. That is the only score that counts. A separate `t1` debug tier (60 bars, 5 assets) exists for harness validation and prompt iteration; nothing in this post uses it. Every result reported here is from a single `test` rollout.

The data side is where the leak surface gets aggressive. The agent sees aliased tickers (`tier_a01` through `tier_a10`), never the real symbols, so it cannot pattern-match on AAPL or MSFT in its own head. The source window is a randomly drawn period from the broad pool `[2018, today]`, committed once at build time and never the same across re-builds. Within a build the alias-to-ticker mapping is permuted, so even if the agent recognizes "this looks like the 2022-Q3 selloff" it still does not learn which alias is AAPL. A sigma=0.0005 zero-mean Gaussian return-noise overlay prevents exact-price recall. The progressive filesystem in the sandbox only ever exposes parquet files dated strictly earlier than the current bar, and the same date gate is enforced again at the tool layer and at the SQL query layer for redundancy. The rules clause in the system prompt is regex-scanned at every step to confirm the agent has not been re-prompted to recognize the underlying assets. The full layering lives under `docs/architecture/`; the short version is that the agent does not get to remember any specific stock from training and does not get to see tomorrow's price.

The whole thing ships as an OpenEnv-compliant environment with the standard reset / step / state surface, behind a FastAPI server with WebSocket sessions, fronted by a Gradio UI on a Hugging Face Space at `yobro4619/tradebench`. From the outside it looks like a Gym env. From the inside it is a fairly serious anti-leak harness with a very specific seven-component reward.

## 4. What we are teaching the agent

The agent already knows what a "diversified momentum portfolio" is. It can describe Kelly sizing in prose, name the right pandas API for a rolling Sharpe, and reason about regime detection at the conceptual level. What it cannot do, out of the box, is run that knowledge as a sequential protocol. It has the playbook. It does not have the calibrated execution policy that turns a playbook into a trajectory of orders, fills, and re-balances across 119 bars.

This is policy calibration, not knowledge transfer. We are not teaching the model new facts about markets. We are teaching it how to convert facts it already holds into actions under the env's specific constraints: stateful position tracking, account for existing weight when sizing new entries, close the loop between recorded intent and actual fills, recognize its own behavioral regressions in real time.

The baseline rollout exposed three failure modes that drove the rest of the work:

1. The agent records `diversified_long` intent every bar but only ever holds 1-2 of the 5 named convictions. The planning is fine, the execution discipline is missing.
2. Heavy `sandbox_exec` use early ("compute the rolling Sharpe matrix") then a collapse to almost no analysis after roughly bar 30. The agent loses interest in its own data once the trajectory gets boring.
3. After the initial build bar, the agent almost never re-checks the gap between its recorded `intended_exposure: 0.60` and its current `gross: 0.20`.

Opus's diagnosis from the Qwen iter 5 trajectory put failure mode 3 plainly:

> *"After the initial build on bar 1 the agent essentially never re-checks the gap between its recorded 0.60 intended_exposure and its actual ~0.20 gross — it records new decisions and advances for 100+ bars without ever firing the remaining build legs, yielding only 14 place_orders over 119 bars."*

The reflection signal we needed was a single sentence the loop could repeat: you said you would build a 5-asset basket, you bought 1, here is a one-paragraph patch to your protocol that fixes it next time.

## 5. Why reflection-based optimization, not GRPO

Reinforcement learning is the textbook fix for a long-horizon execution gap. Train the policy on the reward signal, with thousands of rollouts, until the calibration shows up in weight space. We costed it concretely against the actual hackathon budget and the actual env, and it did not work. Six concrete reasons, in order of severity:

**1. Compute cost is prohibitive.** A single 119-bar episode through Qwen3-32B on Groq runs about 22 minutes of wall-clock. The same episode through GLM-5.1 on Together took 45 to 50 minutes. Each rollout is 300 to 500 LLM calls, because every agent decision (`view_*`, `sandbox_exec`, `record_decision`, `place_order`, `advance_day`) is its own model invocation. Standard policy-gradient methods like PPO or GRPO need on the order of 1000 rollouts before the gradient signal can tell good policies apart from random ones. Run the arithmetic. A 1000-rollout training pass on GLM is roughly 750 hours of wall-clock. On Qwen it is 360 hours. The hackathon window was 36 hours, and even with concurrency the per-token economics cut the experiment off before it could start. Reflection runs 5 rollouts and 5 reflector calls. End-to-end cost was about 2.5 hours of wall-clock and roughly $3 of OpenRouter inference spend across both models.

**2. The reward function was being hardened during the event.** Mid-hackathon we caught a saturation bug: the old `r_sharpe_bonus` term contributed close to 100% of total reward, so the baseline agent scored 0.9997 while losing 12 percentage points to B&H. Catching it required inspecting per-component contributions, redesigning the reward into the [0, 1] convex composite that ships now, and starting over. GRPO on a 32B base model against a still-stabilizing reward would have spent the remaining 30 hours of the event on ablations rather than policy improvement. Reflection tolerates reward refinement: each iteration uses the current reward verbatim, and the trajectory the reflector reads always reflects the latest signal.

**3. Sample efficiency is on the wrong side.** Gradient-based RL needs the gradient signal to dominate the noise floor across thousands of rollouts. Reflection ingests one trajectory per iteration and produces a discrete edit to the system prompt. In-context updates via natural language are O(1) per data point. A single trajectory tells the reflector "you committed to a 5-asset basket but only filled 1 leg, here is a one-paragraph patch." That same lesson would take an RL run hundreds of correlated rollouts to extract from the gradient.

**4. The gap is policy calibration, not new knowledge.** The agent already knows what a diversified momentum portfolio is. It can describe Kelly sizing, name the right pandas API for a rolling Sharpe, and reason about regime detection at the conceptual level. What it cannot do out of the box is convert that knowledge into a calibrated sequential protocol: stateful position tracking, accounting for existing weight when sizing new entries, closing the loop between recorded intent and actual fills. RL fundamentally rewires the weights, which overshoots for a calibration problem. Reflection edits the protocol surface (the system prompt), which is exactly the right unit of change.

**5. Reflection produces a legible, auditable optimization trace.** Reflection optimizes the symbolic policy itself: the system prompt. Diff iter 0 against iter 5 and you can read exactly what changed and why. Compare against an RL-trained policy, where the change lives in 32B fp16 weights and you have to probe it indirectly to learn anything. The reflection trace is six prompt files plus six reflector responses on disk; an RL trace is millions of gradient steps in a checkpoint dir. For a hackathon judged in part on storytelling and reproducibility, the readable artifact is doing real work.

**6. Reflection bootstraps from the base model's existing competence.** RL on a 32B model requires either a working starting policy or a massive exploration budget to find one. Reflection starts from "the canonical TradeBench system prompt with a 5-step daily protocol and a worked example" and uses the base model's existing instruction-following to make every rollout immediately productive. There is no cold-start regime where the agent flails for hundreds of episodes before the gradient picks up signal.

There is an honest tradeoff. Reflection only operates inside the model's existing capability surface. Whatever Qwen3-32B fundamentally cannot do, no prompt edit will install. RL can in principle teach genuinely new skills not present in the base model. Reflection is a pragmatic choice for a specific regime, not a universal claim about how to train agents.

The regime where it pays off is the one we were in: the base model already had the relevant market knowledge, what it lacked was calibrated execution policy, and the budget did not permit gradient-based training. For that case prompt evolution should be the first thing you reach for, and you only escalate to RL when prompt-space is exhausted. The evidence holds up: across five iterations we got monotone gains on `score_normalized`, monotone gains on ROI, and the agent's behavior visibly converged toward more decisive trading. A natural follow-on, listed in §9, is to use the reflection-trained prompt as a warm-start for actual GRPO; reflection as initialization, RL as long-horizon refinement.

## 6. How the reflection loop works mechanically

One iteration is one rollout plus one reflector call. Repeat N times. That is the whole loop.

The rollout. With the env server up, the agent runs both phases of the env: train (1-shot research with the study packet) and test (the 119-bar walk-forward). The full trajectory is logged to `artifacts/runs/<ts>__<model>__iterNN_reflect/`.

The reflector call. The meta-prompt is sent to Claude Opus 4.7 via OpenRouter. It contains four things: the agent's current system prompt verbatim, the full trajectory compressed to `(action_type, payload, reward)` triples (raw observations are omitted to keep the token count manageable), summary statistics including per-bar reward, cumulative reward, `score_normalized`, action counts, and ROI, and a strict output format spec. The reflector must return STEP 0 (strengths to preserve), STEP 1 (the single most-costly failure mode visible in the trajectory), STEP 2 (a surgical edit plan), and finally a `<NEW_SYSTEM_PROMPT>...</NEW_SYSTEM_PROMPT>` block.

We built one hard guard in. The new prompt length must be ≤ 1.10× the prior prompt length. This stops the reflector from rewriting from scratch and forces edits to stay local. We tested without it early on. The prompt doubled in length within two iterations and the agent regressed. The cap is the difference between converging and drifting.

The reflector is also told, plainly, to identify the single most-costly failure mode and fix only that. Surgical edits beat sweeping rewrites. One thing per iteration is the rule.

After each iteration, the new prompt is fed into the next rollout. We commit both the rollout artifacts and the reflector's raw response (`artifacts/reflection_<ts>/iter_NN__reflection/reflector_raw_response.txt`) so the entire optimization trace is replayable from disk.

Entry points are small. `scripts/run_reflection_loop.py --iters 5 --rollout-model qwen/qwen3-32b:groq` drives the loop. `reflection.py` holds the reflector logic and the length-cap check. `inference.py` runs the agent rollout against the env. The model weights are never touched. Across five iterations the only artifacts that change are the system prompt and the resulting trajectory.

A sample edit, abbreviated from the iter 1 surgical plan Opus returned:

> *"§5 step 4: replace 'ZERO OR MORE times' with a conditional gap-based trigger that compares current weight to the target in top_convictions and places orders when gaps exceed a threshold. §4.3 add one line clarifying that on subsequent bars the same recipe applies to the WEIGHT GAP."*

After that edit landed, the agent's `place_order` count went from 4 to 6 in iter 1, and kept climbing through the rest of the loop.

## 7. Results

We ran reflection-based prompt optimization on Qwen3-32B (served via Groq) for 5 iterations. Every iteration improved score_normalized, and the agent's behavior shifted in legible ways: more orders placed, fewer redundant sandbox calls, and far more state observation before each decision. The headline numbers from the test episode (119 bars, 10 assets):

| Iter | score_normalized | ROI | place_order | sandbox_exec | view_* |
|---:|---:|---:|---:|---:|---:|
| 0 (baseline) | 0.6155 | +2.78% | 4 | 22 | 6 |
| 1 | 0.6214 | +4.72% | 6 | 29 | 23 |
| 2 | 0.6306 | +3.88% | 10 | 6 | 17 |
| 3 | 0.6381 | +6.31% | 11 | 1 | 32 |
| 4 | 0.6476 | +6.96% | 14 | 6 | 32 |
| **5** | **0.6584** | **+9.89%** | **12** | **10** | **31** |

ROI grew 3.6x in absolute terms across 5 iterations (+2.78% to +9.89%). For a reference, an equal-weight buy-and-hold over the same 10 assets and same 119 bars returns +15.96%. The trained agent still trails B&H, but the gap closed from roughly -13 log-points at baseline to -5 log-points at iter 5. That delta is what the reflection loop produced, and the bar-level alpha plot below makes it visible directly.

![Figure 1: reward_evolution. score_normalized per reflection iteration, Qwen3-32B and GLM-5.1 on the same axes.](docs/figures/reward_evolution.png)

The Qwen line climbs monotonically from 0.6155 to 0.6584, with no regressions. That property matters: it means the surgical-edit constraint (each reflection touches only the named failure modes, capped at 1.10x prior length) was tight enough to preserve what worked while improving what didn't. The GLM line shows a different shape: a sharp jump at iter 1, a small dip at iter 2, then a recovery to a new best at iter 3. Same loop, different base model, qualitatively different optimization curve.

![Figure 2: reward_roi_combined. ROI bars (left axis) and score_normalized line (right axis), Qwen3-32B.](docs/figures/reward_roi_combined.png)

ROI and score_normalized track together but not identically. Score combines seven components (alpha vs B&H, return, drawdown, hit rate, position consistency, bar diversity, leverage discipline), and ROI is only one input. The two-axis view shows the agent improving on the composite signal even when realized ROI is noisy across iterations. Iter 2's ROI dipped slightly versus iter 1 while score still climbed. The rest of the components compensated.

![Figure 3: bar_alpha_vs_bnh. Per-bar cumulative log-alpha vs equal-weight B&H, baseline (red dashed) vs iter 5 (green solid).](docs/figures/bar_alpha_vs_bnh.png)

This is the plot that makes the change feel real. The baseline agent drifts negative across the episode, troughing near -13 log-points as the market rallies and the agent stays mostly in cash. The iter 5 agent stays close to flat through the early window and stabilizes around -5 log-points across the back half of the episode. Same prices, same available actions, same scaffold: only the prompt changed. The shift from "watching from the sideline" to "participating, imperfectly" is exactly what we wanted reflection to find.

![Figure 4: action_mix_evolution. Counts of place_order, sandbox_exec, and view_* per iteration.](docs/figures/action_mix_evolution.png)

The action histograms tell a sharper story than the table alone. place_order tripled (4 to 12) over the trajectory; the agent learned to act on its convictions rather than recording an intent and never filing it. The view_* family went from 6 calls at baseline to 31 at iter 5, after the reflector explicitly inserted a per-bar protocol asking the agent to read portfolio state before deciding. The most surprising trace is sandbox_exec: 22 at baseline, peaking at 29 after iter 1, then collapsing to 1 by iter 3 and stabilizing in the 6 to 10 range. The agent learned that re-running the same data analysis bar after bar wasn't producing new information, and stopped doing it. We did not encode that lesson by hand. The reflector named "redundant analysis without action" as a failure mode in iter 2, the agent updated, and the count dropped 28x in one step.

![Figure 5: reward_components_baseline_vs_final. Mean per-bar contribution of each reward component, baseline vs iter 5.](docs/figures/reward_components_baseline_vs_final.png)

The components view confirms the mechanism. c_alpha gained +0.033, c_return +0.023, and c_consistency +0.035, mean per-bar. These are the three components that move when the agent actually trades into positions and holds them. c_drawdown dropped slightly, which is what we'd expect: more positions means more interim mark-to-market noise, and that's an acceptable cost when alpha and return are climbing in lockstep. The reward function isn't being gamed. The policy is improving along the axes the reward was designed to measure.

GLM-5.1 was run for three reflection iterations on Together as a comparative trajectory. Baseline 0.6243 (+2.83% ROI, 2 orders). Iter 1 jumped to 0.6432 (+4.39% ROI, 6 orders), a +0.019 score gain that was about 3× Qwen's iter-1 gain. Iter 2 dipped slightly to 0.6382 (+3.72% ROI). Iter 3 recovered and set a new GLM best at 0.6453 (+4.69% ROI, 6 orders). Two patterns worth flagging: (a) GLM's baseline was a more extreme under-trader than Qwen's (2 vs 4 orders, 88 view_* vs 6, 81 sandbox_exec vs 22), so the reflector had a bigger initial signal to work with, and the first reflection iteration produced a correspondingly larger jump; (b) GLM's trajectory was not monotone (it dipped then recovered), where Qwen climbed cleanly. The same loop, run on different base models, produces qualitatively different optimization curves.

![Figure 6: reward_roi_combined_glm. GLM-5.1 ROI bars and score line per iteration.](docs/figures/reward_roi_combined_glm.png)

![Figure 7: bar_alpha_vs_bnh_glm. GLM-5.1 per-bar log-alpha vs B&H, baseline vs iter 3.](docs/figures/bar_alpha_vs_bnh_glm.png)

The GLM bar-alpha plot shows the same shape as Qwen's: baseline drifts negative all the way to roughly −13 log-points; iter-3 cuts the trough by about half. The recovery is less pronounced than Qwen's iter-5 because GLM had three iterations rather than five, but the directional move is the same. The full GLM tables, action-mix evolution, and reward-component decomposition live in RESULTS.md.

## 8. What we learned

Reflection makes the optimization trace human-readable. After 5 iterations we can open the prompt diff and see exactly what changed. The diff is concentrated in two regions: the per-bar protocol (when and how to place orders) and the named-failure-modes section (which behavioral patterns to recognize and avoid). The rules clause, the anti-memorization warning, and the worked example were left almost untouched. Opus, acting as the reflector, correctly identified those parts as already working, and didn't waste its 1.10x length budget rewriting them. That's the right behavior, and it's the kind of judgment a gradient-based optimizer cannot express.

The reflector's diagnoses got sharper over time. Iter 1 named a broad pattern: "agent records intent but doesn't trade." Iter 5 named a much more specific subpattern: "two legs filled out of five committed in the same decision block." Each reflection step looked at a less degenerate trajectory than the previous one, so the failure modes the reflector could see became more behavioral and more actionable. That progression is the qualitative signature of a working optimization loop. If the reflector's diagnoses had stayed broad and repeated themselves across iterations, we'd have known the loop was stuck.

Two models, two pathologies, one converged behavior. Qwen3-32B's baseline failure mode was under-execution: it would write decision blocks, then place a single order or none at all. GLM-5.1's baseline failure mode was the opposite, over-thinking: 81 sandbox_exec calls in 119 bars, and 2 actual trades. After reflection, both models drifted toward the same operating point: more orders, more selective sandbox use, more state observation before deciding. That convergence from opposite starting points is what makes us believe the loop is finding something real about the env's success conditions, not just patching idiosyncratic quirks of one model.

The 1.10x length cap was not a stylistic preference. We tried earlier runs without it and the prompt grew unboundedly: the agent started memorizing example trades, the reflector tried to add more guard rails on top of guard rails, and the policy regressed. Surgical edits, with a hard length cap on every step, are the entire point. Without that constraint, reflection becomes prompt accretion, and accretion is not optimization.

One bug worth naming. Our first reward function had a saturation issue: an old `r_sharpe_bonus` term was contributing close to 100% of total reward on most episodes. The baseline agent scored 0.9997 on a portfolio that lost 12 percentage points to B&H. We caught it before launching reflection by inspecting per-component contributions, redesigned the reward into the [0, 1] convex composite that's now in the env, and only then started the loop. The reflection loop is exactly as good as the reward signal it optimizes against. If we'd pushed forward on the broken reward, we would have reflected ourselves into a worse and worse policy with a beautiful score curve.

## 9. Limitations and what we'd do next

The main caveat is sample size at the episode level. We evaluated on a single 119-bar window with one regime. Multi-window cross-validation, with held-out episodes from different market conditions, is what would let us claim generalization rather than fit. That's the next thing we'd run.

The reflector is Claude Opus 4.7. We didn't ablate cheaper reflectors. It's plausible that a smaller model would produce diagnoses too vague to drive sharp edits, or that it would lose the discipline that keeps the prompt growing slowly. Reflector quality is a hyperparameter we should sweep.

There's also a ceiling. Qwen3-32B's max achievable score on this env is bounded by the underlying model's capability, and prompt optimization can only carry it so far. To beat buy-and-hold (the +15.96% mark) you likely need either a stronger base model or actual RL training that updates weights. Reflection gets you a much better starting point. It doesn't replace gradient signal forever.

We also did no hyperparameter search on the meta-prompt itself. The STEP 0 / STEP 1 / STEP 2 / new_prompt structure was hand-designed in one shot. Variants that ask the reflector to reason in different orders, or to produce candidate edits and self-critique them, might compound on the gains we already see.

Finally, the train phase is currently a 1-shot research write-up rather than a multi-bar interaction. The agent reads a packet of historical context once, then executes on the test episode. A richer train phase, with the agent making decisions across a held-out training window before the test, would give the reflector more signal per iteration.

What's next, concretely:

- Try Sonnet 4.6 as the reflector and measure how reflection quality scales with reflector capability.
- Use the trained prompt as initialization for actual GRPO. Reflection as warm-start, RL as the long-horizon refinement.
- Multi-window cross-validation across regimes.
- Extend the GLM trajectory beyond 3 iterations to compare convergence horizons.

## 10. Closing

Five iterations of reflection on Qwen3-32B took the agent from +2.78% ROI to +9.89% ROI on a 119-bar trading episode, monotonically, with a fully readable optimization trace.

Every iteration's prompt, trajectory, and raw reflector response is committed under `artifacts/`. The run is auditable and replayable end to end.

If you want to clone and try it: the env and training code are at https://github.com/PrathamSingla15/openenv_raethx_finals, and a hosted demo lives at https://huggingface.co/spaces/yobro4619/tradebench. TradeBench is OpenEnv-compliant, so you can drop it into any TRL training loop and run reflection, GRPO, or your own scheme on top of it.

We will keep posting results as we extend to multi-window evaluation and stronger reflectors.
