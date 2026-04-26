# Metric panel and reward decomposition

TradeBench scores agents on a **7-component composite reward** computed by
`tradebench.rewards.composite.compute_composite_reward`. The primary
signal is dense log-wealth; six bounded regularizers shape behavior
without dominating. This page documents the reward, the auxiliary
episode summary, and the verifier output.

## Composite reward (per-bar)

Every `advance_day` call emits a `RewardBreakdown` whose seven fields
flow through `TradeObservation.reward_breakdown`:

| Component | Formula | Range |
|---|---|---|
| `r_wealth` | `log(V_after / V_before)` | unbounded; typical ±0.001–0.05 |
| `r_sharpe_bonus` | `clip(annualized_sharpe(recent_returns) / 2, 0, 0.2)`, zero if `n < 5` or sample variance ≤ 0 | `[0, 0.2]` |
| `r_drawdown` | `clip(-((hwm - V_after)/hwm)², -0.3, 0)`, zero above HWM | `[-0.3, 0]` |
| `r_turnover` | `clip(-(turnover - 0.5)⁺ × 0.2, -0.1, 0)` | `[-0.1, 0]` |
| `r_concentration` | `clip(-(hhi - 0.5)⁺ × 0.2, -0.1, 0)` | `[-0.1, 0]` |
| `r_rules` | `-1.0` if `check_rules_clause` fired during this bar else `0.0` | `{-1, 0}` |
| `r_hack` | `-1.0` if `scan_forbidden_globals` fired during this bar else `0.0` | `{-1, 0}` |

Aggregations:

- **Telemetry scalar**: `r_wealth + 0.5 × Σ secondary` (returned as
  `tool_output.reward` from `advance_day`). Used for plots and human
  inspection only.
- **GRPO training path**: TRL's `GRPOTrainer(reward_funcs=[wealth_fn,
  sharpe_fn, drawdown_fn, turnover_fn, concentration_fn, rules_fn,
  hack_fn])` consumes the per-component vector — group-relative
  advantages are computed per signal and combined inside TRL. Rationale
  in `IMPROVEMENT_PLAN.md` §4.3.

A worked numerical example for one trading day lives in `IMPROVEMENT_PLAN.md` §4.4.

## Episode summary metrics

`tradebench.scoring.metrics.score_episode(...)` returns these top-level
fields, derived purely from the ledger event log:

- `initial_value`, `final_value`
- `score` / `cumulative_log_wealth`: `log(final / initial)`
- `max_drawdown`: worst peak-to-trough on the value path
- `realized_volatility`, `sharpe`, `sortino`
- `turnover`: sum of per-session traded-notional / lagged-value ratios
- `concentration`: mean end-of-day Herfindahl index of risky-sleeve weights
- `total_costs`, `cost_drag = total_costs / initial_cash`
- `step_rewards`: one `log(V_t1 / V_t0)` value per `DayAdvanced`
- `sessions`: per-session detail including exposure, turnover, costs, and
  realized position weights

These flow into the human-readable Gradio panel at `/web/`.

## Verifier output

`python -m tradebench.verifier --tier t1 --seed 42` runs the
conformance / leak / replay / determinism battery and prints a PASS/FAIL
table:

```
TradeBench verifier — tier=t1 seed=42
  PASS  conformance.observation       all 10 observations valid
  PASS  conformance.action_rejection  malformed actions rejected with structured errors
  PASS  leak.fs[step=10]              inspected 9 files, no future partitions
  PASS  leak.no_future_fit            no preprocessors with fit_through_date > current_date
  PASS  replay                        all components match across 5 steps; max r_wealth delta=0.000e+00
  PASS  determinism                   |ΔV| = 0.000e+00 < 1e-06; 11 events; 10 actions
RESULT: PASS
```

Exit code 0 iff every check is PASS or WARN; 1 on any FAIL. See
`src/tradebench/verifier/` for each check's implementation; full spec in
`IMPROVEMENT_PLAN.md` §6.

## Interpretation

- Rank by `cumulative_log_wealth` across the chosen tier (T1 / T2 / T3).
- Read `turnover` and `cost_drag` together when a strategy looks too
  active — both should rise together if the trades are real.
- Per-component reward curves (one series per `r_*` field) are the right
  diagnostic when total reward stagnates: which signals improved, which
  regressed.
- Any `r_rules == -1.0` or `r_hack == -1.0` is a fatal soft kill — the
  episode is not invalid but the trained policy is being told this
  behavior must stop.
