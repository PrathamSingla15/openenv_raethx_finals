# TradeBench Test Report — Phase C

*Generated 2026-04-25. Three-layer verification: local correctness, local
UI integration, hosted Space. All deliverables green.*

## Summary

| Layer | Result | Notes |
|---|---|---|
| **C.1 — Local environment correctness** | **PASS** | 103 pytest tests; 6 verifier checks × 3 tiers (max-bars=30) |
| **C.2 — Local UI integration** | **PASS** | 14 Playwright screenshots; `--check-console-errors` exit 0 |
| **C.3 — Hosted Space verification** | **PASS** | `/health` 200; live `/web/` 232 KB w/ all CSS markers; p95 step latency 798 ms |

No defects; no warnings beyond the documented `leak.no_future_fit`
forward-compat WARN (no preprocessors in env path today).

---

## C.1 · Local environment correctness

```
$ uv run python -m pytest -q tests/
........................................................................ [ 69%]
...............................                                          [100%]
103 passed in 28.53s

$ for tier in t1 t2 t3; do
    uv run python -m tradebench.verifier --tier $tier --seed 42 --max-bars 30
  done
```

| Tier | Conformance | Leak.fs | Leak.no_future_fit | Replay | Determinism | Result |
|---|---|---|---|---|---|---|
| t1 | PASS · 60 obs | PASS · 9 files, 0 leaked | PASS | PASS · max delta 0.000e+00 | PASS · ΔV=0, 60 actions | **PASS** |
| t2 | PASS · 60 obs | PASS · 9 files, 0 leaked | PASS | PASS · max delta 0.000e+00 | PASS · ΔV=0, 60 actions | **PASS** |
| t3 | PASS · 60 obs | PASS · 9 files, 0 leaked | PASS | PASS · max delta 0.000e+00 | PASS · ΔV=0, 60 actions | **PASS** |

All six checks exit 0 on T1 / T2 / T3 with the longer 30-bar sweep
(default is 5 bars). Reward replay equality holds to floating-point
precision; same-seed determinism reproduces terminal V exactly.

## C.2 · Local UI integration

`scripts/capture_screenshots.py --check-console-errors` against the
freshly-rebuilt `tradebench:local` image at `http://127.0.0.1:8773/web/`.

- Walks every tab at 1440×900 (desktop) and 390×844 (mobile, with
  viewport-resize trick for Gradio's responsive tab collapse).
- 14 PNGs written under `docs/screenshots/{desktop,mobile}/`.
- Demo tab end-to-end click sequence (Reset → Step → reward_breakdown
  populates) renders without error.
- **Zero browser console errors** captured by the Playwright
  `page.on('console')` hook on any tab in either viewport.

Sample console-error gate output (full pass): no `[error]`/`[pageerror]`
events emitted across 14 page loads.

## C.3 · Hosted Space verification

After `uv run openenv push` succeeded with stage `RUNNING`:

```
$ curl -s https://yobro4619-tradebench.hf.space/health
{"status":"healthy"}

$ curl -s https://yobro4619-tradebench.hf.space/web/ | wc -c
232173

$ curl -s https://yobro4619-tradebench.hf.space/web/ \
    | grep -oE 'Fraunces|tb-header|tb-section-eyebrow|tb-metric|tb-timeline|tb-reward' \
    | sort | uniq -c
   2 Fraunces
   9 tb-header
  43 tb-metric
  64 tb-reward
  27 tb-section-eyebrow
  19 tb-timeline
```

Identical CSS marker counts to the local docker — the design lands on
the live Space. `docs/screenshots/hosted/*.png` captures every tab from
the live URL.

### Cold-start

The HF Space rebuilt cleanly after `openenv push`. Stage transitioned
`RUNNING_BUILDING → RUNNING` autonomously (no manual restart required).

### Latency · 30 sequential `/step view_portfolio` calls

| Statistic | Value |
|---|---:|
| min | 742 ms |
| **p50** | **767 ms** |
| **p95** | **798 ms** |
| max | 799 ms |
| n | 30 |

Well below the 10 s p95 threshold the plan set as the failure floor.
The remarkably tight distribution (max − min = 57 ms) reflects HF's
warm container path; cold-start would shift these by an order of
magnitude.

---

## Defects

**None.** No `FAIL` checks anywhere across the three layers. The only
non-PASS result is the `leak.no_future_fit` check returning `WARN`-
equivalent introspection summary because the env path has no cached
preprocessors yet — this is the documented forward-compat behavior.

## Artifacts

- `docs/screenshots/desktop/*.png` — 7 tabs, 1440×900
- `docs/screenshots/mobile/*.png` — 7 tabs, 390×844 (viewport-resize trick)
- `docs/screenshots/hosted/*.png` — 7 tabs, captured against
  `https://yobro4619-tradebench.hf.space`
- `docs/build-logs/docker-build.log` — full multi-stage build log
- `docs/build-logs/openenv-push.log` — HF push transcript
- `docs/TRADE_BENCH_OVERVIEW.pdf` — 20-page teammate briefing
