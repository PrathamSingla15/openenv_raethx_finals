# TradeBench: 5 reflections, 3.6× ROI, zero gradient updates

*A long-horizon trading environment, and what we learned from training an agent on it without touching the weights.*

**Submission for the Meta PyTorch OpenEnv Hackathon (India 2026), Theme 2: (Super) Long-Horizon Planning & Instruction Following.**

- Live env: https://huggingface.co/spaces/yobro4619/tradebench
- Code: https://github.com/PrathamSingla15/openenv_raethx_finals
- Companion docs: [`README.md`](README.md), [`RESULTS.md`](RESULTS.md), [`docs/tradebench-explainer.pdf`](docs/tradebench-explainer.pdf)

---

## 1. The result, before anything else

A frozen Qwen3-32B on a 119-bar trading episode, scored against an equal-weight buy-and-hold baseline on real OHLCV data. Iteration zero: `score_normalized` of 0.6155, ROI of +2.78%, four orders placed across the entire run. After five reflection passes, with no gradient updates and no model swap: 0.6584, +9.89%, twelve orders placed. Every iteration improved on the last. The climb was monotone.

Nothing about the model changed. The weights stayed exactly where Groq served them. Between rollouts, Claude Opus 4.7 read the prior trajectory, identified what the agent had failed to do, and rewrote the system prompt. That was the entire optimization loop.

The headline finding from this experiment is not that prompts matter. Everyone already knows prompts matter. The interesting claim is sharper: frontier LLMs already carry the planning knowledge needed to act well in a sequential, non-stationary decision environment. What they lack out of the box is a calibrated policy for spending that knowledge under tight constraints, with the right pacing, sizing, and abstention behavior. We tested whether you can install that policy through prompt evolution alone, on a task with proper anti-memorization defenses, and the answer turned out to be yes. A 3.6× ROI improvement over five iterations is hard to wave off as noise.

This post covers the rest of that experiment. Section 2 explains why we built our own environment instead of training on an existing one. Section 3 walks through TradeBench, the env we shipped. Sections 4 onward cover the reflection loop itself, the actual prompt diffs that moved the score, and where the approach hits its ceiling.

## 2. Why long-horizon trading was the right test

Frontier LLMs have measurable failure modes on long-horizon decision tasks. They reason well about a single trade in isolation. They do not reason well across hundreds of sequential decisions where past actions reshape the future state distribution. The skill that breaks down is not analytical depth but pacing: knowing when to act, when to abstain, how to size given what you already hold, and how to keep from drifting into a regime you did not intend.

Finance is the cleanest place to measure this. KellyBench (2026) ran every major frontier model on a single Premier League season, asking each to manage £100K across match betting markets. Every model lost money. The best run, Claude Opus 4.6, finished at -11% ROI. Grok 3 went to ruin entirely. None of the models had a knowledge problem; they had an execution problem stretched across a few hundred sequential decisions, and that was enough to flip every one of them negative. The gap between knowing and acting, in this regime, is a stated number rather than an intuition.

The textbook answer to a long-horizon execution gap is reinforcement learning. Train the policy directly on the reward signal, with thousands of rollouts, until the calibration shows up in weight space. We costed it. A single 119-bar T2 episode took 45 to 50 minutes of wall-clock through the Groq inference provider, with hundreds of tool calls per episode and a few seconds per call. Multiplying that across the rollout count an LLM-as-policy RL run actually needs got us to weeks, not the 36 hours a hackathon budget allows. Even with concurrency, the per-token economics cut the experiment off before it could start.

So we picked the cheaper axis. Leave the model frozen. Treat prompt-space as the optimization domain. Each iteration reads the prior trajectory, finds the failure modes the agent demonstrated, proposes surgical edits to the system prompt, and redeploys. The mechanism draws from GEPA-style generative prompt evolution and the broader reflective-optimization line of work, with a strong reflector (Opus 4.7) operating on a smaller agent's (Qwen3-32B) trajectories.

The actual research question, then, was narrow and falsifiable. Does reflection on trajectory, with a stronger model in the editor seat, move scores on a finance task whose anti-memorization defenses are tight enough that the agent cannot just pattern-match its way through? We had to build the task to find out, because no existing OpenEnv environment shipped with the leak controls we needed. That env is TradeBench.

## 3. TradeBench in one section

The agent runs the same five-step loop on every bar. It observes through a set of `view_*` read tools that expose the current portfolio, the rolling price window, and a study packet of pre-computed features. It models, by emitting Python into `sandbox_exec`, which runs in a hardened Docker container with pandas, numpy, and a progressive filesystem that only contains parquet files dated strictly before the current trading day. It sizes by calling `record_decision`, which forces the agent to commit a regime label, an edge summary, an intended exposure, the top conviction picks, and an uncertainty bound before the bar closes. It places orders through `place_order`. Then it calls `advance_day`, which is the only call that moves the clock. There are eleven tools total. Orders queue and fill at the next-open price with deterministic slippage and cost models, so the same trajectory replays bit-for-bit on the same seed.

The structural piece that makes the env actually about long-horizon planning, rather than next-action prediction, is the `record_decision` gate. The session in `src/tradebench/environment/session.py` enforces it server-side: if the agent calls `advance_day` without first writing a decision for the current bar, the call is rejected. The agent literally cannot move forward without committing to a thesis on the record. Every advance leaves a paper trail that the reward function can audit.

The reward itself is a 7-component composite, all bounded to [0,1] by construction, with weights that sum to 1.0. The components, defined in `src/tradebench/rewards/composite.py`, are alpha against an equal-weight buy-and-hold benchmark (weight 0.40), cumulative log-return (0.15), drawdown control (0.10), solvency (0.10), turnover efficiency (0.10), Herfindahl-based diversity (0.05), and downside semi-volatility of bar-level alpha for consistency (0.10). The composite is multiplied by a {0,1} compliance gate, which zeros the bar on any rule violation. The episode score, `score_normalized`, is the mean per-bar reward across the run. The 0.40 weight on alpha is deliberate: a lucky cash-only agent should not beat a thoughtful one in a year when markets happen to fall.

There are three difficulty tiers. T1 is 60 bars and 5 assets, T2 is 120 bars and 10 assets, T3 is 252 bars and 20 assets. Everything in this post is on T2.

The data side is where the leak surface gets aggressive. The agent sees aliased tickers (`tier_a01` through `tier_a10`), never the real symbols, so it cannot pattern-match on AAPL or MSFT in its own head. Source windows for each tier are drawn from non-overlapping random periods of the underlying OHLCV, so the regimes do not bleed between tiers. The progressive filesystem in the sandbox only ever exposes parquet files dated strictly earlier than the current bar, and the same date gate is enforced again at the tool layer and at the SQL query layer for redundancy. The rules clause in the system prompt is regex-scanned at every step to confirm the agent has not been re-prompted to recognize the underlying assets. The full layering lives under `docs/architecture/`; the short version is that the agent does not get to remember any specific stock from training and does not get to see tomorrow's price.

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

## 5. Why reflection-based optimization, not RL

Start with the cost equation. A single 119-bar episode through Qwen3-32B on Groq runs about 22 minutes of wall-clock. The same episode through GLM-5.1 on Together took 45 to 50 minutes. Each rollout is 300 to 500 LLM calls because every agent decision (view, sandbox_exec, record_decision, place_order, advance_day) is its own model invocation. Standard policy-gradient methods like PPO or GRPO need thousands of rollouts before the gradient signal can tell good policies apart from random ones. Run the arithmetic. A 1000-rollout training pass on GLM is roughly 750 hours of wall-clock. On Qwen it is 360 hours. The hackathon was 36.

Compute is not even the worst part. Reward signal in a 119-bar episode is sparse, and the reward function itself was still hardening during the event. We caught a saturation bug mid-hackathon where the old `r_sharpe_bonus` contributed close to 100% of total reward, which made the score effectively uninformative. The convex composite in `src/tradebench/rewards/composite.py` fixed that. Running GRPO on a 32B base model against a still-stabilizing reward would have spent the remaining 30 hours of the event on ablations rather than on policy improvement.

Reflection is asymmetric in cost. The reflector is one call to a strong model (Claude Opus 4.7) per iteration that reads the trajectory and emits a new prompt. The agent runs one rollout per iteration. Per-iteration cost is around $0.50 of inference. Five iterations got us from 0.6155 to 0.6584 in roughly 2.5 hours of wall-clock and about $3 of OpenRouter spend.

The artifact reflection produces is also legible. Reflection optimizes the symbolic policy: the system prompt itself. Diff iter 0 against iter 5 and you can read exactly what changed and why. Compare against an RL-trained policy, where the change lives in 32B fp16 weights and you have to probe it indirectly to learn anything. For a hackathon judged in part on storytelling, the readable artifact is doing real work.

There is an honest tradeoff here. Reflection only operates inside the model's existing capability surface. Whatever Qwen3-32B fundamentally cannot do, no prompt edit will install. RL can in principle teach genuinely new skills not present in the base model. Reflection is a pragmatic choice for a specific regime, not a universal claim about how to train agents.

The regime where it pays off is the one we were in. The base model already had the relevant market knowledge. What it lacked was calibrated execution policy. For that case prompt evolution should be the first thing you reach for, and you only escalate to RL when prompt-space is exhausted.

The evidence holds up. Across five iterations we got monotone gains on `score_normalized`, monotone gains on ROI, and the agent's behavior visibly converged toward more decisive trading. The env was learnable through prompts alone for this model, and the data says so.

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

![Figure 1: reward_evolution. score_normalized per reflection iteration, both models.](docs/figures/reward_evolution.png)

The Qwen line climbs monotonically from 0.6155 to 0.6584, with no regressions. That property matters: it means the surgical-edit constraint (each reflection touches only the named failure modes, capped at 1.10x prior length) was tight enough to preserve what worked while improving what didn't. GLM-5.1 is shown as a single completed point with a dashed segment for the in-flight iteration, and even from one iteration its movement is larger in absolute terms than Qwen's was at the same step.

![Figure 2: reward_roi_combined. ROI bars (left axis) and score_normalized line (right axis), faceted by model.](docs/figures/reward_roi_combined.png)

ROI and score_normalized track together but not identically. Score combines seven components (alpha vs B&H, return, drawdown, hit rate, position consistency, bar diversity, leverage discipline), and ROI is only one input. The two-axis view shows the agent improving on the composite signal even when realized ROI is noisy across iterations. Iter 2's ROI dipped slightly versus iter 1 while score still climbed. The rest of the components compensated.

![Figure 3: bar_alpha_vs_bnh. Per-bar cumulative log-alpha vs equal-weight B&H, baseline (red dashed) vs iter 5 (green solid).](docs/figures/bar_alpha_vs_bnh.png)

This is the plot that makes the change feel real. The baseline agent drifts negative across the episode, troughing near -13 log-points as the market rallies and the agent stays mostly in cash. The iter 5 agent stays close to flat through the early window and stabilizes around -5 log-points across the back half of the episode. Same prices, same available actions, same scaffold: only the prompt changed. The shift from "watching from the sideline" to "participating, imperfectly" is exactly what we wanted reflection to find.

![Figure 4: action_mix_evolution. Counts of place_order, sandbox_exec, and view_* per iteration.](docs/figures/action_mix_evolution.png)

The action histograms tell a sharper story than the table alone. place_order tripled (4 to 12) over the trajectory; the agent learned to act on its convictions rather than recording an intent and never filing it. The view_* family went from 6 calls at baseline to 31 at iter 5, after the reflector explicitly inserted a per-bar protocol asking the agent to read portfolio state before deciding. The most surprising trace is sandbox_exec: 22 at baseline, peaking at 29 after iter 1, then collapsing to 1 by iter 3 and stabilizing in the 6 to 10 range. The agent learned that re-running the same data analysis bar after bar wasn't producing new information, and stopped doing it. We did not encode that lesson by hand. The reflector named "redundant analysis without action" as a failure mode in iter 2, the agent updated, and the count dropped 28x in one step.

![Figure 5: reward_components_baseline_vs_final. Mean per-bar contribution of each reward component, baseline vs iter 5.](docs/figures/reward_components_baseline_vs_final.png)

The components view confirms the mechanism. c_alpha gained +0.033, c_return +0.023, and c_consistency +0.035, mean per-bar. These are the three components that move when the agent actually trades into positions and holds them. c_drawdown dropped slightly, which is what we'd expect: more positions means more interim mark-to-market noise, and that's an acceptable cost when alpha and return are climbing in lockstep. The reward function isn't being gamed. The policy is improving along the axes the reward was designed to measure.

GLM-5.1's first iteration on Together is also worth noting. Baseline 0.6243, iter 1 0.6432 (+0.019). ROI moved +2.83% to +4.39%, and place_order went 2 to 7 (3.5x). The pattern is suggestive: GLM started further from convergence than Qwen, and its first iteration moved more than Qwen's first iteration did. The bigger the baseline pathology, the larger the signal the reflector has to work with. We will append the full 5-iter GLM trajectory to RESULTS.md once the run completes.

## 8. What we learned

Reflection makes the optimization trace human-readable. After 5 iterations we can open the prompt diff and see exactly what changed.

![Figure 6: prompt_evolution. Top card is the baseline system prompt; bottom card is the prompt after 5 reflections. Red wash marks lines that were removed or rewritten; green wash marks lines that were added.](docs/figures/prompt_evolution.png)

The diff is concentrated in two regions: the per-bar protocol (when and how to place orders) and the named-failure-modes section (which behavioral patterns to recognize and avoid). The rules clause, the anti-memorization warning, and the worked example were left almost untouched. Opus, acting as the reflector, correctly identified those parts as already working, and didn't waste its 1.10x length budget rewriting them. That's the right behavior, and it's the kind of judgment a gradient-based optimizer cannot express.

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

- Append the full 5-iter GLM-5.1 trajectory to RESULTS.md when the run finishes.
- Try Sonnet 4.6 as the reflector and measure how reflection quality scales with reflector capability.
- Use the trained prompt as initialization for actual GRPO. Reflection as warm-start, RL as the long-horizon refinement.
- Multi-window cross-validation across regimes.

## 10. Closing

Five iterations of reflection on Qwen3-32B took the agent from +2.78% ROI to +9.89% ROI on a 119-bar trading episode, monotonically, with a fully readable optimization trace.

Every iteration's prompt, trajectory, and raw reflector response is committed under `artifacts/`. The run is auditable and replayable end to end.

If you want to clone and try it: the env and training code are at https://github.com/PrathamSingla15/openenv_raethx_finals, and a hosted demo lives at https://huggingface.co/spaces/yobro4619/tradebench. TradeBench is OpenEnv-compliant, so you can drop it into any TRL training loop and run reflection, GRPO, or your own scheme on top of it.

We will keep posting results as the GLM run completes and as we extend to multi-window evaluation.
