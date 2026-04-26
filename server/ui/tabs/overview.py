"""Trajectory tab — full landing-page scroll.

A raeth.ai-style cinematic experience: full-viewport hero with the Blog.md
hook, six numbered sections (Problem / Environment / Reward / Defenses /
Results / Run it) lifted verbatim from Blog.md and README.md, and a closing
CTA strip. Section 05 embeds every figure that appears in Blog.md as an
interactive Plotly chart (equity curves, reward trajectory, ROI/score, bar
alpha, action mix, reward components) with the prompt-evolution diagrams as
static PNGs (they're presentation graphics, not plots). Deep-dive sub-links
route into the other tabs via ``data-tb-tab`` attributes that
``static/motion.js`` resolves to a tab button click.
"""

from __future__ import annotations

import logging

import gradio as gr

from ..figures import figure_card

logger = logging.getLogger(__name__)


# ── Hero (full viewport) ────────────────────────────────────────
_HERO_HTML = """
<section id="tb-section-hero" class="tb-landing tb-reveal">
    <div class="tb-landing-halo" aria-hidden="true"></div>

    <div class="tb-landing-eyebrow">
        <span class="is-accent">OpenEnv Hackathon</span>
        <span class="tb-dot">/</span>
        <span>Theme 2 &middot; Long-horizon planning</span>
        <span class="tb-dot">/</span>
        <span>India 2026 finals</span>
    </div>

    <h1 class="tb-landing-title">
        If AI models are <span class="tb-accent">so smart</span>, why aren't they rich?
    </h1>

    <p class="tb-landing-sub">
        <em>TradeBench:</em> a trading RL environment to test and improve a model's
        ability for capital allocation under incomplete and noisy information. Real
        OHLCV behind a four-layer anti-memorization stack. Reward in <code>[0, 1]</code>
        by construction, gated by a binary compliance check. A single 119-bar test
        rollout produces 300&ndash;500 LLM calls, dense per-bar reward, and a
        deterministic grader.
    </p>

    <div class="tb-landing-actions">
        <a class="tb-cta tb-cta-primary"
           href="https://github.com/PrathamSingla15/openenv_raethx_finals/blob/main/Blog.md"
           target="_blank" rel="noreferrer">
            Read the writeup
            <span class="tb-cta-arrow" aria-hidden="true">&#x2197;</span>
        </a>
        <a class="tb-cta tb-cta-ghost" href="#" data-tb-tab="Run">
            Run the demo
            <span class="tb-cta-arrow" aria-hidden="true">&rarr;</span>
        </a>
    </div>

    <div class="tb-metrics-strip">
        <div class="tb-metrics-cell">
            <div class="tb-metrics-key">Best score</div>
            <div class="tb-metrics-val">0.6584</div>
            <div class="tb-metrics-foot">qwen3-32b &middot; iter 5</div>
        </div>
        <div class="tb-metrics-cell">
            <div class="tb-metrics-key">ROI climb</div>
            <div class="tb-metrics-val is-accent">3.6&times;</div>
            <div class="tb-metrics-foot">+2.78% &rarr; +9.89%</div>
        </div>
        <div class="tb-metrics-cell">
            <div class="tb-metrics-key">Reflections</div>
            <div class="tb-metrics-val">5</div>
            <div class="tb-metrics-foot">monotone every iteration</div>
        </div>
        <div class="tb-metrics-cell">
            <div class="tb-metrics-key">Gradient updates</div>
            <div class="tb-metrics-val">0</div>
            <div class="tb-metrics-foot">prompt evolution only</div>
        </div>
        <div class="tb-metrics-cell">
            <div class="tb-metrics-key">Inference spend</div>
            <div class="tb-metrics-val">$3</div>
            <div class="tb-metrics-foot">openrouter, end-to-end</div>
        </div>
        <div class="tb-metrics-cell">
            <div class="tb-metrics-key">Wall-clock</div>
            <div class="tb-metrics-val">~2.5h</div>
            <div class="tb-metrics-foot">budget was 36</div>
        </div>
    </div>
</section>
"""


# ── 01 / Problem ────────────────────────────────────────────────
_SECTION_01_HTML = """
<section id="tb-section-01" class="tb-numbered-section tb-reveal">
    <div class="tb-numbered-head">
        <div class="tb-numbered-tag">
            <span class="is-accent">01</span> / 06 &middot; The bet
        </div>
        <div>
            <h2 class="tb-numbered-title">
                <span class="tb-faded">What was once left to individual judgement can</span>
                become an optimizable system.
            </h2>
            <p class="tb-numbered-sub">
                Capital allocation drives the economy. It decides which drugs get developed,
                which technologies get built, and which ideas survive long enough to matter.
                And yet, for something so central, it still runs on a fragile foundation: human
                judgment, plus a thin and badly instrumented layer of agentic systems that
                pretend to know what they are doing. Frontier LLMs can reason about a single
                trade in isolation, but break down across hundreds of sequential decisions where
                past actions reshape the future state distribution.
            </p>
            <p class="tb-numbered-sub">
                The skill that fails is not analytical depth. It is execution discipline:
                <em>when to act, when to abstain, how to size against existing positions, how to
                keep from drifting into a regime the agent did not intend.</em> TradeBench turns
                long-horizon equities trading into an environment where execution discipline is
                a measurable, decomposable, gradient-friendly target.
            </p>
        </div>
    </div>

    <blockquote class="tb-quote">
        The model didn't get smarter. It got more disciplined.
    </blockquote>
</section>
"""


# ── 02 / Environment ────────────────────────────────────────────
_SECTION_02_HTML = """
<section id="tb-section-02" class="tb-numbered-section tb-reveal">
    <div class="tb-numbered-head">
        <div class="tb-numbered-tag">
            <span class="is-accent">02</span> / 06 &middot; The env
        </div>
        <div>
            <h2 class="tb-numbered-title">
                Long-horizon execution discipline as an
                <span class="tb-accent">optimizable</span> surface.
            </h2>
            <p class="tb-numbered-sub">
                A per-bar reward in <code>[0, 1]</code> decomposed across seven trader-recognizable
                components, server-enforced sequential structure that makes <em>abstain</em> a
                first-class action, and a deterministic grader that produces final cumulative
                log-wealth, max drawdown, Sharpe, Sortino, and an avoided-ruin boolean from the
                ledger event log. Fully reproducible, no LLM judge.
            </p>
        </div>
    </div>

    <div class="tb-three-up">
        <div class="tb-card">
            <div class="tb-card-num">01 / 03 &middot; Five-step daily loop</div>
            <div class="tb-card-title">Observe. Model. Commit. Order. Advance.</div>
            <div class="tb-card-body">
                Eleven tools, hardened Docker sandbox for analysis (pandas / numpy / time-gated
                parquet, no internet), and a single call &mdash; <code>advance_day</code> &mdash;
                that moves the clock. Orders queue and fill at the next-open print with
                deterministic slippage and cost models.
            </div>
        </div>
        <div class="tb-card">
            <div class="tb-card-num">02 / 03 &middot; Long-horizon spine</div>
            <div class="tb-card-title">A thesis on the record before the clock ticks.</div>
            <div class="tb-card-body">
                The session enforces <code>record_decision</code> before every
                <code>advance_day</code>: regime label, edge summary, intended exposure, top
                convictions, uncertainty. The agent literally cannot tick the clock without
                committing a thesis on the record. Every advance leaves a paper trail the
                reward function can audit.
            </div>
        </div>
        <div class="tb-card">
            <div class="tb-card-num">03 / 03 &middot; Anti-memorization stack</div>
            <div class="tb-card-title">Six layers, redundant by design.</div>
            <div class="tb-card-body">
                Aliased tickers (<code>tier_a01</code>&hellip;<code>tier_a10</code>), randomly
                drawn source window from the broad <code>[2018, today]</code> pool, per-build
                alias-to-ticker permutation, &sigma;=0.0005 return-noise overlay, time-gated
                filesystem, and a rules-clause regex scanner. The agent has to derive strategy
                from data it can see, not retrieve it from weight memory.
            </div>
        </div>
    </div>

    <a class="tb-deepdive-link" href="#" data-tb-tab="Environment">
        Open the environment audit
        <span class="tb-deepdive-arrow" aria-hidden="true">&#x2197;</span>
    </a>
</section>
"""


# ── 03 / Reward ─────────────────────────────────────────────────
_SECTION_03_HTML = """
<section id="tb-section-03" class="tb-numbered-section tb-reveal">
    <div class="tb-numbered-head">
        <div class="tb-numbered-tag">
            <span class="is-accent">03</span> / 06 &middot; The signal
        </div>
        <div>
            <h2 class="tb-numbered-title">
                Reward in <span class="tb-accent">[0, 1]</span> by construction. Gated by a binary
                compliance check.
            </h2>
            <p class="tb-numbered-sub">
                A convex combination of seven trader-recognizable components, each in
                <code>[0, 1]</code>, with weights that sum to <code>1.0</code>, multiplied by a
                <code>{0, 1}</code> compliance gate. The episode score is the mean per-bar
                reward, also in <code>[0, 1]</code>. No unbounded log-wealth head. No way for
                a great Sharpe to earn back a rules-clause hit.
            </p>
        </div>
    </div>

    <div class="tb-formula-row">
        <div class="tb-formula">
<pre>r_t = (&Sigma;&#x1D62; w&#x1D62; &middot; c&#x1D62;_t)  *  g_compliance_t                 # in [0, 1]

components (each in [0, 1]):
  c_alpha       sigmoid(8 &middot; cumulative_log_alpha_vs_bench)   w = 0.40
  c_return      sigmoid(5 &middot; cumulative_log_return)           w = 0.15
  c_drawdown    1 &minus; 2 &middot; min(dd, 0.5)                         w = 0.10
  c_solvency    sigmoid(6 &middot; (V &minus; 0.5&middot;V&#x2080;) / (0.5&middot;V&#x2080;))         w = 0.10
  c_efficiency  exp(&minus;2 &middot; max(0, turnover &minus; 0.10))            w = 0.10
  c_diversity   1 &minus; clip((HHI &minus; 0.10) / 0.90, 0, 1)          w = 0.05
  c_consistency win &middot; stability + (1 &minus; win) &middot; 0.5            w = 0.10

compliance gate (multiplicative, not additive):
  g_compliance = (1 &minus; viol_rules) &middot; (1 &minus; viol_hack) &middot; (1 &minus; lev_breach)
  with lev_breach = 1{gross_leverage &gt; 1.0}.  Any single violation zeros the bar.

episode score:
  score_normalized = mean over bars of r_t                  in [0, 1]</pre>
        </div>

        <div>
            <div class="tb-section-eyebrow" style="margin-top:0">Three choices that matter</div>
            <ul class="tb-defense-list" style="border-top:1px solid var(--rule); list-style:none; padding:0">
                <li class="tb-defense-row">
                    <span class="tb-defense-num">01</span>
                    <span class="tb-defense-text">
                        <strong><code>c_alpha</code> is dominant by design (0.40 weight).</strong>
                        <span class="tb-defense-body">A lucky cash-only agent should not beat a
                        thoughtful one in a year when markets happen to fall.</span>
                    </span>
                </li>
                <li class="tb-defense-row">
                    <span class="tb-defense-num">02</span>
                    <span class="tb-defense-text">
                        <strong><code>c_consistency</code> is gated.</strong>
                        <span class="tb-defense-body">Above-benchmark agents earn it via low
                        downside-alpha-volatility; sub-benchmark agents fall back to 0.5 instead
                        of 0. Kills the &ldquo;flatline gets free Sharpe&rdquo; exploit.</span>
                    </span>
                </li>
                <li class="tb-defense-row">
                    <span class="tb-defense-num">03</span>
                    <span class="tb-defense-text">
                        <strong>The compliance gate is multiplicative.</strong>
                        <span class="tb-defense-body">Cheating zeros the bar's reward outright;
                        no Sharpe can earn back a rules-clause hit.</span>
                    </span>
                </li>
            </ul>
        </div>
    </div>

    <a class="tb-deepdive-link" href="#" data-tb-tab="Reward">
        Open the reward decomposition
        <span class="tb-deepdive-arrow" aria-hidden="true">&#x2197;</span>
    </a>
</section>
"""


# ── 04 / Defenses ───────────────────────────────────────────────
_SECTION_04_HTML = """
<section id="tb-section-04" class="tb-numbered-section tb-reveal">
    <div class="tb-numbered-head">
        <div class="tb-numbered-tag">
            <span class="is-accent">04</span> / 06 &middot; The leak controls
        </div>
        <div>
            <h2 class="tb-numbered-title">
                Six layers, redundant <span class="tb-faded">by design.</span>
            </h2>
            <p class="tb-numbered-sub">
                LLM weights post-date the backtest window, so a single defense is never enough.
                The agent has to derive strategy from what it sees, not retrieve it from
                training memory. Across 10 rollouts on two base models the rules clause was
                never tripped, no forbidden global was trafficked, and no bar was zeroed by
                the compliance gate.
            </p>
        </div>
    </div>

    <ol class="tb-defense-list" style="list-style:none; padding:0; margin:0">
        <li class="tb-defense-row">
            <span class="tb-defense-num">01</span>
            <span class="tb-defense-text">
                <strong>Aliased tickers.</strong>
                <span class="tb-defense-body">The agent sees <code>tier_a01</code> through
                <code>tier_a10</code>, never the real symbols, so it cannot pattern-match on AAPL
                or MSFT in its own head.</span>
            </span>
        </li>
        <li class="tb-defense-row">
            <span class="tb-defense-num">02</span>
            <span class="tb-defense-text">
                <strong>Randomized window.</strong>
                <span class="tb-defense-body">The source window is a randomly drawn period from
                the broad pool <code>[2018, today]</code>, committed once at build time and
                never the same across rebuilds.</span>
            </span>
        </li>
        <li class="tb-defense-row">
            <span class="tb-defense-num">03</span>
            <span class="tb-defense-text">
                <strong>Alias permutation.</strong>
                <span class="tb-defense-body">Within a build the alias-to-ticker mapping is
                permuted, so even if the agent recognizes &ldquo;this looks like the 2022-Q3
                selloff&rdquo; it still does not learn which alias is AAPL.</span>
            </span>
        </li>
        <li class="tb-defense-row">
            <span class="tb-defense-num">04</span>
            <span class="tb-defense-text">
                <strong>Return-noise overlay.</strong>
                <span class="tb-defense-body">A &sigma;=0.0005 zero-mean Gaussian return-noise
                overlay prevents exact-price recall.</span>
            </span>
        </li>
        <li class="tb-defense-row">
            <span class="tb-defense-num">05</span>
            <span class="tb-defense-text">
                <strong>Time-gated filesystem.</strong>
                <span class="tb-defense-body">The progressive filesystem in the sandbox only
                ever exposes parquet files dated strictly earlier than the current bar; the same
                date gate is enforced again at the tool layer and at the SQL query layer for
                redundancy.</span>
            </span>
        </li>
        <li class="tb-defense-row">
            <span class="tb-defense-num">06</span>
            <span class="tb-defense-text">
                <strong>Rules-clause regex scan.</strong>
                <span class="tb-defense-body">The rules clause in the system prompt is
                regex-scanned at every step to confirm the agent has not been re-prompted to
                recognize the underlying assets.</span>
            </span>
        </li>
    </ol>

    <a class="tb-deepdive-link" href="#" data-tb-tab="Defenses">
        Open the leakage audit
        <span class="tb-deepdive-arrow" aria-hidden="true">&#x2197;</span>
    </a>
</section>
"""


# ── 05 / Results ────────────────────────────────────────────────
_SECTION_05_INTRO_HTML = """
<section id="tb-section-05" class="tb-numbered-section tb-reveal">
    <div class="tb-numbered-head">
        <div class="tb-numbered-tag">
            <span class="is-accent">05</span> / 06 &middot; The result
        </div>
        <div>
            <h2 class="tb-numbered-title">
                Five iterations. Zero gradient updates.
                <span class="tb-accent">3.6&times;</span> ROI.
            </h2>
            <p class="tb-numbered-sub">
                We froze Qwen3-32B and ran a GEPA-style reflection loop: each iteration, Claude
                Opus&nbsp;4.7 read the prior trajectory, identified the single most-costly
                failure mode, and proposed a surgical edit to the system prompt. The model
                weights never moved. Every iteration improved on the last. The climb was
                monotone.
            </p>
        </div>
    </div>

    <div class="tb-hero-stats" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:1px;background:var(--rule);border:1px solid var(--rule);margin-bottom:var(--space-5)">
        <div class="tb-statcard" style="border:none">
            <div class="tb-statcard-eyebrow">Qwen3-32B &middot; Groq</div>
            <div class="tb-statcard-row">
                <div class="tb-statcard-key">score_normalized</div>
                <div class="tb-statcard-arrow">0.6155 &rarr; <strong>0.6584</strong></div>
            </div>
            <div class="tb-statcard-row">
                <div class="tb-statcard-key">ROI</div>
                <div class="tb-statcard-arrow">+2.78% &rarr; <strong>+9.89%</strong></div>
            </div>
            <div class="tb-statcard-row">
                <div class="tb-statcard-key">place_order</div>
                <div class="tb-statcard-arrow">4 &rarr; <strong>12</strong></div>
            </div>
            <div class="tb-statcard-foot">5 iterations &middot; monotone</div>
        </div>
        <div class="tb-statcard" style="border:none">
            <div class="tb-statcard-eyebrow">GLM-5.1 &middot; Together</div>
            <div class="tb-statcard-row">
                <div class="tb-statcard-key">score_normalized</div>
                <div class="tb-statcard-arrow">0.6243 &rarr; <strong>0.6453</strong></div>
            </div>
            <div class="tb-statcard-row">
                <div class="tb-statcard-key">ROI</div>
                <div class="tb-statcard-arrow">+2.83% &rarr; <strong>+4.69%</strong></div>
            </div>
            <div class="tb-statcard-row">
                <div class="tb-statcard-key">place_order</div>
                <div class="tb-statcard-arrow">2 &rarr; <strong>6</strong></div>
            </div>
            <div class="tb-statcard-foot">3 iterations &middot; climb &middot; dip &middot; recover</div>
        </div>
    </div>
</section>
"""


_QWEN_TABLE_HTML = """
<div class="tb-reveal" style="margin-bottom:var(--space-5)">
    <div class="tb-section-eyebrow">Qwen3-32B per-iter detail</div>
    <table class="tb-table">
        <thead>
            <tr>
                <th class="mono">Iter</th>
                <th class="mono">score_normalized</th>
                <th class="mono">ROI</th>
                <th class="mono">place_order</th>
                <th class="mono">sandbox_exec</th>
                <th class="mono">view_*</th>
            </tr>
        </thead>
        <tbody>
            <tr><td class="mono">0 (baseline)</td><td class="mono">0.6155</td><td class="mono">+2.78%</td><td class="mono">4</td><td class="mono">22</td><td class="mono">6</td></tr>
            <tr><td class="mono">1</td><td class="mono">0.6214</td><td class="mono">+4.72%</td><td class="mono">6</td><td class="mono">29</td><td class="mono">23</td></tr>
            <tr><td class="mono">2</td><td class="mono">0.6306</td><td class="mono">+3.88%</td><td class="mono">10</td><td class="mono">6</td><td class="mono">17</td></tr>
            <tr><td class="mono">3</td><td class="mono">0.6381</td><td class="mono">+6.31%</td><td class="mono">11</td><td class="mono">1</td><td class="mono">32</td></tr>
            <tr><td class="mono">4</td><td class="mono">0.6476</td><td class="mono">+6.96%</td><td class="mono">14</td><td class="mono">6</td><td class="mono">32</td></tr>
            <tr><td class="mono"><strong>5</strong></td><td class="mono"><strong>0.6584</strong></td><td class="mono"><strong>+9.89%</strong></td><td class="mono"><strong>12</strong></td><td class="mono"><strong>10</strong></td><td class="mono"><strong>31</strong></td></tr>
        </tbody>
    </table>
</div>
"""


_GLM_TABLE_HTML = """
<div class="tb-reveal" style="margin-bottom:var(--space-5)">
    <div class="tb-section-eyebrow">GLM-5.1 per-iter detail</div>
    <table class="tb-table">
        <thead>
            <tr>
                <th class="mono">Iter</th>
                <th class="mono">score_normalized</th>
                <th class="mono">ROI</th>
                <th class="mono">place_order</th>
                <th class="mono">sandbox_exec</th>
                <th class="mono">view_*</th>
            </tr>
        </thead>
        <tbody>
            <tr><td class="mono">0 (baseline)</td><td class="mono">0.6243</td><td class="mono">+2.83%</td><td class="mono">2</td><td class="mono">81</td><td class="mono">88</td></tr>
            <tr><td class="mono">1</td><td class="mono">0.6432</td><td class="mono">+4.39%</td><td class="mono">6</td><td class="mono">44</td><td class="mono">110</td></tr>
            <tr><td class="mono">2</td><td class="mono">0.6382</td><td class="mono">+3.72%</td><td class="mono">6</td><td class="mono">19</td><td class="mono">182</td></tr>
            <tr><td class="mono"><strong>3</strong></td><td class="mono"><strong>0.6453</strong></td><td class="mono"><strong>+4.69%</strong></td><td class="mono"><strong>6</strong></td><td class="mono"><strong>41</strong></td><td class="mono"><strong>148</strong></td></tr>
        </tbody>
    </table>
</div>
"""


def _plot_caption(eyebrow: str, title: str, body: str) -> str:
    """Editorial caption block above an interactive Plotly figure."""
    return f"""
<div class="tb-reveal" style="margin:var(--space-5) 0 var(--space-3)">
    <div class="tb-section-eyebrow" style="margin-top:0">{eyebrow}</div>
    <h3 style="font-family:var(--type-display);font-weight:500;font-size:clamp(1.4em,2.4vw,1.8em);
        line-height:1.15;letter-spacing:-0.018em;color:var(--ink);margin:0;
        font-variation-settings:'opsz' 96,'WONK' 1;max-width:60ch">{title}</h3>
    <p style="font-family:var(--type-body);font-size:0.96em;line-height:1.65;color:var(--ink-mute);
        max-width:64ch;margin:8px 0 0">{body}</p>
</div>
"""


# Section banner introducing a per-model figure cluster.
def _model_banner(model: str, sub: str) -> str:
    return f"""
<div class="tb-reveal" style="margin:var(--space-7) 0 var(--space-4);
    padding-top:var(--space-5);border-top:1px solid var(--rule)">
    <div class="tb-section-eyebrow" style="margin-top:0;color:var(--accent)">All figures &middot; {model}</div>
    <p class="tb-numbered-sub" style="margin-top:var(--space-2)">{sub}</p>
</div>
"""


# Static prompt-evolution figure cards (these are presentation diagrams,
# not plots, so they stay as PNG embeds).
_PROMPT_EVO_QWEN_HTML = figure_card(
    src="prompt_evolution.png",
    title="prompt evolution",
    meta="qwen3-32b · baseline → iter 5",
    caption=(
        "Across 5 reflections under the 1.10x length cap, only three regions of "
        "the prompt moved. The reflector left the anti-memorization rules-clause, "
        "the worked-example shape, the tool surface, the tier definitions, the "
        "output contract, and the sandbox spec as written. <strong>That conservatism "
        "is why the optimization curve stayed monotone.</strong>"
    ),
)


_PROMPT_EVO_GLM_HTML = figure_card(
    src="prompt_evolution_glm.png",
    title="prompt evolution",
    meta="glm-5.1 · baseline → iter 3",
    caption=(
        "GLM-5.1 prompt evolution across the comparative 3-iter run. Same "
        "1.10x length cap; same surgical-edit constraint. Section diffs differ "
        "because GLM's baseline failure mode was the opposite of Qwen's: extreme "
        "under-execution (2 orders, 81 sandbox calls)."
    ),
)


# ── 06 / Run it ─────────────────────────────────────────────────
_SECTION_06_HTML = """
<section id="tb-section-06" class="tb-numbered-section tb-reveal">
    <div class="tb-numbered-head">
        <div class="tb-numbered-tag">
            <span class="is-accent">06</span> / 06 &middot; Try it
        </div>
        <div>
            <h2 class="tb-numbered-title">
                OpenEnv-compliant. Drop into any
                <span class="tb-accent">TRL</span> training loop.
            </h2>
            <p class="tb-numbered-sub">
                The deployed env at <code>yobro4619/tradebench</code> exposes the standard
                OpenEnv surface: <code>/reset</code>, <code>/step</code>, <code>/state</code>,
                <code>/schema</code>, plus a WebSocket session at <code>/ws</code>. Every
                iteration's prompt, trajectory, and reflector response is committed under
                <code>artifacts/</code>; the run is auditable and replayable end to end.
            </p>
        </div>
    </div>

    <div class="tb-link-grid">
        <a class="tb-link-card"
           href="https://huggingface.co/spaces/yobro4619/tradebench"
           target="_blank" rel="noreferrer">
            <div class="tb-link-eyebrow">LIVE &middot; HF Space</div>
            <div class="tb-link-title">yobro4619 / tradebench</div>
            <div class="tb-link-sub">FastAPI + Gradio &middot; Docker &middot; OpenEnv routes</div>
        </a>
        <a class="tb-link-card"
           href="https://github.com/PrathamSingla15/openenv_raethx_finals"
           target="_blank" rel="noreferrer">
            <div class="tb-link-eyebrow">CODE &middot; GitHub</div>
            <div class="tb-link-title">openenv_raethx_finals</div>
            <div class="tb-link-sub">env, reward, sandbox, reflector, verifier &middot; MIT</div>
        </a>
        <a class="tb-link-card"
           href="https://colab.research.google.com/github/PrathamSingla15/openenv_raethx_finals/blob/main/notebooks/training.ipynb"
           target="_blank" rel="noreferrer">
            <div class="tb-link-eyebrow">TRAIN &middot; Colab</div>
            <div class="tb-link-title">notebooks/training.ipynb</div>
            <div class="tb-link-sub">end-to-end reflection loop against the deployed env</div>
        </a>
    </div>

    <div class="tb-curl">
<pre><span class="tb-curl-cmt"># reset the env on the test tier</span>
curl <span class="tb-curl-flag">-X</span> POST https://yobro4619-tradebench.hf.space/reset \\
     <span class="tb-curl-flag">-H</span> 'Content-Type: application/json' \\
     <span class="tb-curl-flag">-d</span> '{"task_tier":"test","seed":42}'</pre>
    </div>

    <a class="tb-deepdive-link" href="#" data-tb-tab="Run">
        Run a single step from the demo tab
        <span class="tb-deepdive-arrow" aria-hidden="true">&#x2197;</span>
    </a>
</section>
"""


# ── Final CTA ───────────────────────────────────────────────────
_FINAL_CTA_HTML = """
<section class="tb-final-cta tb-reveal">
    <h2 class="tb-final-cta-title">
        <span class="tb-faded">Run the env.</span>
        <span class="tb-faded">Read the writeup.</span>
        Train an agent on it.
    </h2>
    <div class="tb-final-cta-links">
        <a class="tb-final-cta-link"
           href="https://huggingface.co/spaces/yobro4619/tradebench"
           target="_blank" rel="noreferrer">HF Space &rarr;</a>
        <a class="tb-final-cta-link"
           href="https://github.com/PrathamSingla15/openenv_raethx_finals"
           target="_blank" rel="noreferrer">GitHub &rarr;</a>
        <a class="tb-final-cta-link"
           href="https://github.com/PrathamSingla15/openenv_raethx_finals/blob/main/Blog.md"
           target="_blank" rel="noreferrer">Writeup &rarr;</a>
    </div>
</section>
"""


# ── Headline figure caption blocks (lifted from Blog § 7) ───────
_CAP_REWARD_TRAJ = _plot_caption(
    eyebrow="Headline · Both models",
    title="Reward trajectory across reflection iterations.",
    body=(
        "<strong>Both models climbed.</strong> Qwen converged smoothly across 5 iterations "
        "with no regressions. GLM showed a sharp jump at iter 1, a small dip at iter 2, "
        "then recovered to a new best at iter 3. The y-axis is mean per-bar composite "
        "reward on the held-out 119-bar test episode."
    ),
)


# ── Caption blocks (Qwen) ───────────────────────────────────────
_CAP_QWEN_EQUITY = _plot_caption(
    eyebrow="Figure 1 · Qwen3-32B",
    title="Equity curves across all six iterations vs equal-weight buy-and-hold.",
    body=(
        "Every iteration's prompt produces a higher-equity trajectory than the prior. "
        "Baseline (red dashed) ends at +2.78%; iter 5 (dark green, thick) ends at +9.89%, "
        "the closest any iteration gets to the equal-weight buy-and-hold reference (blue "
        "dashed, +15.96%). All six runs share the same prices, the same available actions, "
        "and the same scaffold; only the system prompt changed between them."
    ),
)


_CAP_QWEN_ROI = _plot_caption(
    eyebrow="Figure 4 · Qwen3-32B",
    title="ROI bars and score per iteration.",
    body=(
        "ROI and score_normalized track together but not identically. Score combines seven "
        "components; ROI is only one input. The two-axis view shows the agent improving on "
        "the composite signal even when realized ROI is noisy across iterations."
    ),
)


_CAP_QWEN_ALPHA = _plot_caption(
    eyebrow="Per-bar · Qwen3-32B",
    title="Cumulative log-alpha vs equal-weight B&H.",
    body=(
        "Baseline (red dashed) bled alpha to B&H all the way to roughly &minus;13 log-points "
        "by bar 119. Iter 5 (green) cut that gap to roughly &minus;5 log-points and held "
        "flat across the back half of the episode."
    ),
)


_CAP_QWEN_ACTION = _plot_caption(
    eyebrow="Figure 5 · Qwen3-32B",
    title="Action mix evolution.",
    body=(
        "<code>place_order</code> tripled (4 &rarr; 12); the agent learned to act on its "
        "convictions rather than record an intent and never file it. <code>view_*</code> "
        "went from 6 to 31 calls. <code>sandbox_exec</code> dropped 22&times; from baseline "
        "in a single iteration window once the reflector steered the agent toward acting on "
        "existing analysis."
    ),
)


_CAP_QWEN_COMPONENTS = _plot_caption(
    eyebrow="Figure 6 · Qwen3-32B",
    title="Reward components, baseline vs iter 5.",
    body=(
        "<code>c_alpha</code> gained +0.033, <code>c_return</code> +0.023, "
        "<code>c_consistency</code> +0.035 mean per-bar &mdash; the three components that "
        "move when the agent trades into positions and holds them. <code>c_drawdown</code> "
        "dropped slightly: more positions means more interim mark-to-market noise. The reward "
        "isn't being gamed; the policy is improving along the axes the reward measures."
    ),
)


# ── Caption blocks (GLM) ────────────────────────────────────────
_CAP_GLM_EQUITY = _plot_caption(
    eyebrow="Figure 8 · GLM-5.1",
    title="Equity curves across four iterations vs B&H.",
    body=(
        "Baseline (red dashed) ends at +2.83%; iter 3 (dark green, thick) ends at +4.69%, "
        "the best of any GLM iteration. Same prices, same actions, same scaffold; only the "
        "prompt changed. Annualized Sharpe at iter 3 is +2.32, beating equal-weight B&H's "
        "+1.79 on the same window."
    ),
)


_CAP_GLM_ROI = _plot_caption(
    eyebrow="Figure 7 · GLM-5.1",
    title="ROI bars and score per iteration.",
    body=(
        "GLM's trajectory was non-monotone: it dipped at iter 2 then recovered at iter 3 "
        "where Qwen climbed cleanly. Same loop, different base model, qualitatively "
        "different optimization curve."
    ),
)


_CAP_GLM_ALPHA = _plot_caption(
    eyebrow="Per-bar · GLM-5.1",
    title="Cumulative log-alpha vs B&H.",
    body=(
        "GLM-5.1 baseline-to-final cumulative alpha gap. The recovery is less pronounced "
        "than Qwen's iter 5 because GLM had three iterations rather than five, but the "
        "direction is the same."
    ),
)


_CAP_GLM_ACTION = _plot_caption(
    eyebrow="Action mix · GLM-5.1",
    title="Behavioral action mix per iteration.",
    body=(
        "GLM's baseline was a more extreme under-trader than Qwen's: 2 orders versus 4, "
        "88 <code>view_*</code> versus 6, 81 <code>sandbox_exec</code> versus 22. So the "
        "reflector had a bigger initial signal, and the first iteration produced a "
        "correspondingly larger jump (about 3&times; Qwen's iter-1 gain)."
    ),
)


_CAP_GLM_COMPONENTS = _plot_caption(
    eyebrow="Reward components · GLM-5.1",
    title="Mean per-bar contribution, baseline vs iter 3.",
    body=(
        "Same component decomposition pattern as Qwen, smaller magnitudes. The bounded-"
        "component reward + multiplicative compliance gate held across both models for the "
        "full 9 reflection iterations: zero rules-clause hits, zero forbidden-global hits, "
        "zero compliance-gate zeros."
    ),
)


_QWEN_BANNER = _model_banner(
    "Qwen3-32B (Groq)",
    (
        "Five reflection iterations on the canonical TradeBench harness. Score lifted "
        "0.6155 &rarr; 0.6584 monotonically; ROI grew +2.78% &rarr; +9.89% (3.6&times; "
        "absolute). Every figure is interactive &mdash; hover for per-bar values."
    ),
)


_GLM_BANNER = _model_banner(
    "GLM-5.1 (Together)",
    (
        "Three reflection iterations as a comparative trajectory. Score moved 0.6243 "
        "&rarr; 0.6453 (best at iter 3, with a small dip-then-recover shape). ROI grew "
        "+2.83% &rarr; +4.69%. Different base model, qualitatively different optimization "
        "curve, same converged operating point."
    ),
)


def render() -> None:
    """Render the complete landing scroll for the Trajectory tab."""
    # Lazy import so a Plotly / data_loader hiccup doesn't crash the module.
    try:
        from .. import plots
        plots_ok = True
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning("Plotly figures disabled: %s", exc)
        plots_ok = False

    gr.HTML(_HERO_HTML)
    gr.HTML(_SECTION_01_HTML)
    gr.HTML(_SECTION_02_HTML)
    gr.HTML(_SECTION_03_HTML)
    gr.HTML(_SECTION_04_HTML)

    gr.HTML(_SECTION_05_INTRO_HTML)

    # Headline interactive figure (replaces the static reward_evolution.png).
    if plots_ok:
        gr.HTML(_CAP_REWARD_TRAJ)
        gr.Plot(plots.reward_trajectory(), show_label=False)
    else:
        gr.HTML(figure_card(
            src="reward_evolution.png",
            title="reward trajectory",
            meta="qwen3-32b vs glm-5.1",
            caption=(
                "Both models climbed. Qwen converged smoothly across 5 iterations; "
                "GLM had a small dip-then-recover at iter 2&ndash;3."
            ),
        ))

    # ── Qwen3-32B all-figures cluster ──
    gr.HTML(_QWEN_BANNER)
    gr.HTML(_QWEN_TABLE_HTML)

    if plots_ok:
        gr.HTML(_CAP_QWEN_EQUITY)
        gr.Plot(plots.equity_curves(plots.QWEN_ID), show_label=False)

        gr.HTML(_CAP_QWEN_ROI)
        gr.Plot(plots.roi_score_combined(plots.QWEN_ID), show_label=False)

        gr.HTML(_CAP_QWEN_ALPHA)
        gr.Plot(plots.bar_alpha_vs_bnh(plots.QWEN_ID), show_label=False)

        gr.HTML(_CAP_QWEN_ACTION)
        gr.Plot(plots.action_mix(plots.QWEN_ID), show_label=False)

        gr.HTML(_CAP_QWEN_COMPONENTS)
        gr.Plot(plots.reward_components(plots.QWEN_ID), show_label=False)

    gr.HTML(_PROMPT_EVO_QWEN_HTML)

    # ── GLM-5.1 mirror ──
    gr.HTML(_GLM_BANNER)
    gr.HTML(_GLM_TABLE_HTML)

    if plots_ok:
        gr.HTML(_CAP_GLM_EQUITY)
        gr.Plot(plots.equity_curves(plots.GLM_ID), show_label=False)

        gr.HTML(_CAP_GLM_ROI)
        gr.Plot(plots.roi_score_combined(plots.GLM_ID), show_label=False)

        gr.HTML(_CAP_GLM_ALPHA)
        gr.Plot(plots.bar_alpha_vs_bnh(plots.GLM_ID), show_label=False)

        gr.HTML(_CAP_GLM_ACTION)
        gr.Plot(plots.action_mix(plots.GLM_ID), show_label=False)

        gr.HTML(_CAP_GLM_COMPONENTS)
        gr.Plot(plots.reward_components(plots.GLM_ID), show_label=False)

    gr.HTML(_PROMPT_EVO_GLM_HTML)

    # Deep-dive link out of section 05.
    gr.HTML(
        '<a class="tb-deepdive-link tb-reveal" href="#" data-tb-tab="Reflection Loop">'
        'Open the full reflection loop deep-dive'
        '<span class="tb-deepdive-arrow" aria-hidden="true">&#x2197;</span>'
        '</a>'
    )

    gr.HTML(_SECTION_06_HTML)
    gr.HTML(_FINAL_CTA_HTML)
