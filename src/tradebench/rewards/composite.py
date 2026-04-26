"""Per-bar composite reward in [0, 1] for TradeBench.

Convex combination of seven well-defined components multiplied by a hard
compliance gate. Each component lives in [0, 1] and the seven weights sum
to 1, so the per-step reward is in [0, 1] by construction. The episode
score is the arithmetic mean of per-bar rewards, also in [0, 1].

    r_t = (Sum_i w_i * c_i_t) * g_compliance_t

Components:

    c_alpha       sigmoid(8 * cumulative_log_alpha_vs_bench)         w=0.40
    c_return      sigmoid(5 * cumulative_log_return)                 w=0.15
    c_drawdown    1 - 2 * min(dd, 0.5)                               w=0.10
    c_solvency    sigmoid(6 * (V - 0.5*V0) / (0.5*V0))               w=0.10
    c_efficiency  exp(-2 * max(0, turnover - 0.10))                  w=0.10
    c_diversity   1 - clip((HHI - 0.10) / 0.90, 0, 1)                w=0.05
    c_consistency win * stability + (1 - win) * 0.5                  w=0.10

Compliance gate:

    g_compliance = (1 - viol_rules) * (1 - viol_hack) * (1 - lev_breach)

with lev_breach = 1{gross_leverage > 1.0}. Any single violation zeros
the bar's reward.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from decimal import Decimal

W_ALPHA = 0.40
W_RETURN = 0.15
W_DRAWDOWN = 0.10
W_SOLVENCY = 0.10
W_EFFICIENCY = 0.10
W_DIVERSITY = 0.05
W_CONSISTENCY = 0.10

K_ALPHA = 8.0
K_RETURN = 5.0
K_SOLVENCY = 6.0
K_WIN = 8.0
K_STABILITY = 15.0

DD_FLOOR = 0.5
TURNOVER_THRESHOLD = 0.10
TURNOVER_DECAY = 2.0
HHI_FLOOR = 0.10
HHI_CEILING = 1.00
LEVERAGE_CAP = 1.0
CONSISTENCY_WINDOW = 20


@dataclass(frozen=True)
class RewardBreakdown:
    """Per-bar reward decomposition.

    All ``c_*`` fields are in ``[0, 1]``. ``g_compliance`` is in ``{0, 1}``.
    ``r_total`` is the gated convex combination, also in ``[0, 1]``.
    """

    c_alpha: float
    c_return: float
    c_drawdown: float
    c_solvency: float
    c_efficiency: float
    c_diversity: float
    c_consistency: float
    g_compliance: float
    r_total: float

    def total(self) -> float:
        return self.r_total

    def to_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _clip_unit(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _downside_alpha_semideviation(recent_bar_alphas: Sequence[float]) -> float:
    if not recent_bar_alphas:
        return 0.0
    sq = 0.0
    for a in recent_bar_alphas:
        if a < 0.0:
            sq += a * a
    return math.sqrt(sq / len(recent_bar_alphas))


def _consistency_component(
    cumulative_log_alpha: float,
    recent_bar_alphas: Sequence[float],
) -> float:
    win = _sigmoid(K_WIN * cumulative_log_alpha)
    sigma_down = _downside_alpha_semideviation(recent_bar_alphas)
    stability = math.exp(-K_STABILITY * sigma_down)
    return win * stability + (1.0 - win) * 0.5


def compute_composite_reward(
    *,
    value_after: Decimal | float,
    initial_value: Decimal | float,
    cumulative_log_return: float,
    cumulative_benchmark_log_return: float,
    recent_bar_alphas: Sequence[float],
    high_watermark: Decimal | float,
    turnover_ratio: float,
    hhi: float,
    gross_leverage: float,
    violations_rules: bool,
    violations_hack: bool,
) -> RewardBreakdown:
    """Per-bar reward in [0, 1].

    Parameters
    ----------
    value_after:
        Portfolio market value at end of bar. Used for the solvency gate.
        Must be > 0; bankruptcy is handled upstream.
    initial_value:
        Episode-start portfolio value (``manifest.initial_cash``). Used as the
        anchor for the solvency buffer.
    cumulative_log_return:
        ``log(V_t / V_0)`` for the agent.
    cumulative_benchmark_log_return:
        ``log(B_t / B_0)`` for the equal-weight buy-and-hold benchmark on the
        same universe.
    recent_bar_alphas:
        Sequence of the most recent per-bar alphas (``r_t^p - r_t^b``) for
        the consistency component's downside-semideviation. Up to
        ``CONSISTENCY_WINDOW`` entries; an empty sequence yields zero
        downside-vol (stability = 1).
    high_watermark:
        Maximum portfolio value seen so far.
    turnover_ratio:
        ``gross_traded_notional / value_before`` for this bar.
    hhi:
        Realized Herfindahl of risky-sleeve weights.
    gross_leverage:
        Sum of ``|notional_i| / value_after`` over the risky sleeve. Used to
        detect leverage-cap breaches in the compliance gate.
    violations_rules, violations_hack:
        Booleans drained from the regex scanners on this bar.
    """

    v_after = float(value_after)
    v0 = float(initial_value)
    if v_after <= 0.0 or v0 <= 0.0:
        msg = (
            "portfolio values must be positive to compute composite reward; "
            f"got value_after={v_after!r}, initial_value={v0!r}"
        )
        raise ValueError(msg)

    cum_alpha = float(cumulative_log_return) - float(cumulative_benchmark_log_return)

    c_alpha = _sigmoid(K_ALPHA * cum_alpha)
    c_return = _sigmoid(K_RETURN * float(cumulative_log_return))

    hwm = float(high_watermark)
    if hwm > 0.0 and v_after < hwm:
        dd = (hwm - v_after) / hwm
    else:
        dd = 0.0
    c_drawdown = 1.0 - 2.0 * min(dd, DD_FLOOR)

    ruin_floor = 0.5 * v0
    buffer = (v_after - ruin_floor) / ruin_floor
    c_solvency = _sigmoid(K_SOLVENCY * buffer)

    excess_turnover = max(0.0, float(turnover_ratio) - TURNOVER_THRESHOLD)
    c_efficiency = math.exp(-TURNOVER_DECAY * excess_turnover)

    hhi_excess = (float(hhi) - HHI_FLOOR) / (HHI_CEILING - HHI_FLOOR)
    c_diversity = 1.0 - _clip_unit(hhi_excess)

    c_consistency = _consistency_component(cum_alpha, recent_bar_alphas)

    lev_breach = 1.0 if float(gross_leverage) > LEVERAGE_CAP else 0.0
    g_compliance = (
        (0.0 if violations_rules else 1.0)
        * (0.0 if violations_hack else 1.0)
        * (1.0 - lev_breach)
    )

    convex = (
        W_ALPHA * c_alpha
        + W_RETURN * c_return
        + W_DRAWDOWN * c_drawdown
        + W_SOLVENCY * c_solvency
        + W_EFFICIENCY * c_efficiency
        + W_DIVERSITY * c_diversity
        + W_CONSISTENCY * c_consistency
    )
    r_total = convex * g_compliance

    return RewardBreakdown(
        c_alpha=c_alpha,
        c_return=c_return,
        c_drawdown=c_drawdown,
        c_solvency=c_solvency,
        c_efficiency=c_efficiency,
        c_diversity=c_diversity,
        c_consistency=c_consistency,
        g_compliance=g_compliance,
        r_total=r_total,
    )


__all__ = [
    "RewardBreakdown",
    "compute_composite_reward",
    "W_ALPHA",
    "W_RETURN",
    "W_DRAWDOWN",
    "W_SOLVENCY",
    "W_EFFICIENCY",
    "W_DIVERSITY",
    "W_CONSISTENCY",
]


if __name__ == "__main__":
    failures: list[str] = []

    def _check(name: str, ok: bool, detail: str = "") -> None:
        if ok:
            print(f"PASS  {name}")
        else:
            failures.append(name)
            print(f"FAIL  {name}: {detail}")

    weight_sum = (
        W_ALPHA + W_RETURN + W_DRAWDOWN + W_SOLVENCY
        + W_EFFICIENCY + W_DIVERSITY + W_CONSISTENCY
    )
    _check("weights sum to 1.0", math.isclose(weight_sum, 1.0))

    def _kw(**overrides):
        base = dict(
            value_after=100.0,
            initial_value=100.0,
            cumulative_log_return=0.0,
            cumulative_benchmark_log_return=0.0,
            recent_bar_alphas=(),
            high_watermark=100.0,
            turnover_ratio=0.0,
            hhi=0.10,
            gross_leverage=0.0,
            violations_rules=False,
            violations_hack=False,
        )
        base.update(overrides)
        return base

    rb = compute_composite_reward(**_kw())
    _check("flat-flat c_alpha == 0.5", math.isclose(rb.c_alpha, 0.5, abs_tol=1e-9))
    _check("flat c_return == 0.5", math.isclose(rb.c_return, 0.5, abs_tol=1e-9))
    _check("no DD c_drawdown == 1.0", rb.c_drawdown == 1.0)
    _check("solvency at V0 ≈ 1.0", rb.c_solvency > 0.99)
    _check("low turnover c_efficiency == 1.0", rb.c_efficiency == 1.0)
    _check("HHI at floor c_diversity == 1.0", math.isclose(rb.c_diversity, 1.0))
    _check("g_compliance == 1 when clean", rb.g_compliance == 1.0)
    _check("r_total in [0, 1]", 0.0 <= rb.r_total <= 1.0)

    # do-nothing all-cash, B&H = +16.73%
    bench_log = math.log(1.1673)
    rb = compute_composite_reward(
        **_kw(
            value_after=100.0,
            cumulative_log_return=0.0,
            cumulative_benchmark_log_return=bench_log,
            hhi=0.0,
        ),
    )
    _check(
        "do-nothing all-cash → score in 0.50–0.58",
        0.50 <= rb.r_total <= 0.58,
        detail=f"r_total={rb.r_total:.4f}",
    )

    # broken baseline replay: agent +4.37%, B&H +16.73%
    agent_log = math.log(1.0437)
    rb = compute_composite_reward(
        **_kw(
            value_after=104.37,
            initial_value=100.0,
            cumulative_log_return=agent_log,
            cumulative_benchmark_log_return=bench_log,
            high_watermark=104.37,
            turnover_ratio=0.05,
            hhi=0.20,
        ),
    )
    _check(
        "broken baseline → score in 0.52–0.62",
        0.52 <= rb.r_total <= 0.62,
        detail=f"r_total={rb.r_total:.4f}",
    )

    # equal-weight B&H from bar 1: agent matches benchmark
    rb = compute_composite_reward(
        **_kw(
            value_after=116.73,
            initial_value=100.0,
            cumulative_log_return=bench_log,
            cumulative_benchmark_log_return=bench_log,
            high_watermark=116.73,
            turnover_ratio=0.0,
            hhi=0.10,
        ),
    )
    _check(
        "B&H tied with benchmark → score in 0.65–0.75",
        0.65 <= rb.r_total <= 0.75,
        detail=f"r_total={rb.r_total:.4f}",
    )

    # strong alpha agent: +50% ROI, B&H +16.73%
    strong_log = math.log(1.50)
    rb = compute_composite_reward(
        **_kw(
            value_after=150.0,
            initial_value=100.0,
            cumulative_log_return=strong_log,
            cumulative_benchmark_log_return=bench_log,
            high_watermark=150.0,
            turnover_ratio=0.10,
            hhi=0.15,
        ),
    )
    _check(
        "strong alpha +50% → score >= 0.80",
        rb.r_total >= 0.80,
        detail=f"r_total={rb.r_total:.4f}",
    )

    # losing agent: -10% while B&H +16.73%
    loss_log = math.log(0.90)
    rb = compute_composite_reward(
        **_kw(
            value_after=90.0,
            initial_value=100.0,
            cumulative_log_return=loss_log,
            cumulative_benchmark_log_return=bench_log,
            high_watermark=100.0,
            turnover_ratio=0.20,
            hhi=0.30,
        ),
    )
    _check(
        "loses 10% vs +16% B&H → score <= 0.50",
        rb.r_total <= 0.50,
        detail=f"r_total={rb.r_total:.4f}",
    )

    # rule violation zeroes the bar
    rb = compute_composite_reward(**_kw(violations_rules=True))
    _check("rule violation zeroes r_total", rb.r_total == 0.0)
    _check("rule violation g_compliance == 0", rb.g_compliance == 0.0)

    rb = compute_composite_reward(**_kw(violations_hack=True))
    _check("hack violation zeroes r_total", rb.r_total == 0.0)

    rb = compute_composite_reward(**_kw(gross_leverage=1.5))
    _check("leverage breach zeroes r_total", rb.r_total == 0.0)

    # consistency: sub-benchmark agent gets 0.5 regardless of vol
    rb = compute_composite_reward(
        **_kw(
            cumulative_log_return=-0.10,
            cumulative_benchmark_log_return=0.10,
            recent_bar_alphas=(-0.05, -0.04, -0.06, -0.05),
        ),
    )
    _check(
        "sub-benchmark c_consistency ≈ 0.5",
        abs(rb.c_consistency - 0.5) < 0.05,
        detail=f"c_consistency={rb.c_consistency:.4f}",
    )

    # consistency: above-benchmark + low downside-vol → high
    rb = compute_composite_reward(
        **_kw(
            cumulative_log_return=0.20,
            cumulative_benchmark_log_return=0.10,
            recent_bar_alphas=(0.001, 0.0005, 0.002, 0.0008, 0.0015),
        ),
    )
    _check(
        "above-benchmark low-vol c_consistency > 0.7",
        rb.c_consistency > 0.7,
        detail=f"c_consistency={rb.c_consistency:.4f}",
    )

    # bounds: per-component
    for cum_log in (-2.0, -0.5, 0.0, 0.5, 2.0):
        for bench in (-1.0, 0.0, 1.0):
            for tov in (0.0, 0.5, 5.0):
                for h in (0.0, 0.5, 1.0):
                    rb = compute_composite_reward(
                        **_kw(
                            value_after=max(0.01, 100.0 * math.exp(cum_log)),
                            cumulative_log_return=cum_log,
                            cumulative_benchmark_log_return=bench,
                            turnover_ratio=tov,
                            hhi=h,
                        ),
                    )
                    bounds_ok = all(
                        0.0 <= getattr(rb, name) <= 1.0
                        for name in (
                            "c_alpha",
                            "c_return",
                            "c_drawdown",
                            "c_solvency",
                            "c_efficiency",
                            "c_diversity",
                            "c_consistency",
                            "r_total",
                        )
                    )
                    if not bounds_ok:
                        failures.append(
                            f"bounds@cum={cum_log},bench={bench},tov={tov},hhi={h}"
                        )

    _check(
        "all components stay in [0, 1] across grid",
        not any(name.startswith("bounds@") for name in failures),
    )

    # Decimal acceptance
    rb = compute_composite_reward(
        **_kw(
            value_after=Decimal("104.37"),
            initial_value=Decimal("100"),
            high_watermark=Decimal("104.37"),
            cumulative_log_return=math.log(1.0437),
            cumulative_benchmark_log_return=math.log(1.1673),
        ),
    )
    _check("Decimal inputs accepted", 0.0 <= rb.r_total <= 1.0)

    if failures:
        print(f"\n{len(failures)} FAIL(S): {failures}")
        raise SystemExit(1)
    print("\ncomposite reward OK")
