"""Pytest coverage for the redesigned [0, 1] composite reward.

Mirrors the assertions in ``rewards/composite.py``'s ``__main__`` block and
adds a few targeted regression cases (boundary scenarios, Decimal inputs,
randomized bounds).
"""

from __future__ import annotations

import math
import random
from decimal import Decimal

import pytest

from tradebench.rewards.composite import (
    K_ALPHA,
    K_RETURN,
    RewardBreakdown,
    W_ALPHA,
    W_CONSISTENCY,
    W_DIVERSITY,
    W_DRAWDOWN,
    W_EFFICIENCY,
    W_RETURN,
    W_SOLVENCY,
    compute_composite_reward,
)


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


def test_weights_sum_to_one() -> None:
    total = (
        W_ALPHA + W_RETURN + W_DRAWDOWN + W_SOLVENCY
        + W_EFFICIENCY + W_DIVERSITY + W_CONSISTENCY
    )
    assert math.isclose(total, 1.0)


def test_to_dict_preserves_field_order() -> None:
    rb = RewardBreakdown(
        c_alpha=0.5,
        c_return=0.5,
        c_drawdown=1.0,
        c_solvency=1.0,
        c_efficiency=1.0,
        c_diversity=1.0,
        c_consistency=0.5,
        g_compliance=1.0,
        r_total=0.65,
    )
    assert list(rb.to_dict().keys()) == [
        "c_alpha",
        "c_return",
        "c_drawdown",
        "c_solvency",
        "c_efficiency",
        "c_diversity",
        "c_consistency",
        "g_compliance",
        "r_total",
    ]


def test_flat_flat_pivot() -> None:
    rb = compute_composite_reward(**_kw())
    assert math.isclose(rb.c_alpha, 0.5, abs_tol=1e-9)
    assert math.isclose(rb.c_return, 0.5, abs_tol=1e-9)
    assert rb.c_drawdown == 1.0
    assert rb.c_solvency > 0.99
    assert rb.c_efficiency == 1.0
    assert math.isclose(rb.c_diversity, 1.0)
    assert rb.g_compliance == 1.0
    assert 0.0 <= rb.r_total <= 1.0


def test_alpha_monotonic_in_excess() -> None:
    base = compute_composite_reward(**_kw())
    higher = compute_composite_reward(**_kw(cumulative_log_return=0.10))
    lower = compute_composite_reward(**_kw(cumulative_log_return=-0.10))
    assert lower.c_alpha < base.c_alpha < higher.c_alpha


def test_alpha_pivots_at_zero() -> None:
    rb = compute_composite_reward(
        **_kw(cumulative_log_return=0.5, cumulative_benchmark_log_return=0.5),
    )
    assert math.isclose(rb.c_alpha, 0.5, abs_tol=1e-9)


def test_drawdown_linear_below_hwm() -> None:
    rb = compute_composite_reward(**_kw(value_after=80.0, high_watermark=100.0))
    assert math.isclose(rb.c_drawdown, 1.0 - 2.0 * 0.20)


def test_drawdown_floor_at_50pct() -> None:
    rb = compute_composite_reward(**_kw(value_after=10.0, high_watermark=100.0))
    assert rb.c_drawdown == 0.0


def test_solvency_pivot_at_ruin_floor() -> None:
    rb = compute_composite_reward(**_kw(value_after=50.0, initial_value=100.0))
    assert math.isclose(rb.c_solvency, 0.5, abs_tol=1e-9)


def test_efficiency_decay_above_threshold() -> None:
    low = compute_composite_reward(**_kw(turnover_ratio=0.05))
    mid = compute_composite_reward(**_kw(turnover_ratio=0.50))
    high = compute_composite_reward(**_kw(turnover_ratio=2.00))
    assert low.c_efficiency == 1.0
    assert 0.0 < mid.c_efficiency < 1.0
    assert high.c_efficiency < mid.c_efficiency


def test_diversity_falls_with_concentration() -> None:
    floor = compute_composite_reward(**_kw(hhi=0.10))
    mid = compute_composite_reward(**_kw(hhi=0.55))
    full = compute_composite_reward(**_kw(hhi=1.00))
    assert math.isclose(floor.c_diversity, 1.0)
    assert math.isclose(mid.c_diversity, 0.5, abs_tol=1e-9)
    assert full.c_diversity == 0.0


def test_consistency_falls_back_for_subbenchmark() -> None:
    rb = compute_composite_reward(
        **_kw(
            cumulative_log_return=-0.20,
            cumulative_benchmark_log_return=0.10,
            recent_bar_alphas=(-0.05, -0.03, -0.04),
        ),
    )
    assert abs(rb.c_consistency - 0.5) < 0.05


def test_consistency_high_for_above_benchmark_low_downside() -> None:
    rb = compute_composite_reward(
        **_kw(
            cumulative_log_return=0.20,
            cumulative_benchmark_log_return=0.10,
            recent_bar_alphas=(0.001, 0.002, 0.0005, 0.0015),
        ),
    )
    assert rb.c_consistency > 0.7


def test_compliance_zeroes_total_on_rule_violation() -> None:
    rb = compute_composite_reward(**_kw(violations_rules=True))
    assert rb.g_compliance == 0.0
    assert rb.r_total == 0.0


def test_compliance_zeroes_total_on_hack_violation() -> None:
    rb = compute_composite_reward(**_kw(violations_hack=True))
    assert rb.g_compliance == 0.0
    assert rb.r_total == 0.0


def test_compliance_zeroes_total_on_leverage_breach() -> None:
    rb = compute_composite_reward(**_kw(gross_leverage=1.05))
    assert rb.g_compliance == 0.0
    assert rb.r_total == 0.0


def test_leverage_at_cap_passes() -> None:
    rb = compute_composite_reward(**_kw(gross_leverage=1.0))
    assert rb.g_compliance == 1.0


def test_raises_on_non_positive_value_after() -> None:
    with pytest.raises(ValueError):
        compute_composite_reward(**_kw(value_after=0.0))


def test_raises_on_non_positive_initial_value() -> None:
    with pytest.raises(ValueError):
        compute_composite_reward(**_kw(initial_value=0.0))


def test_decimal_inputs_accepted() -> None:
    rb = compute_composite_reward(
        **_kw(
            value_after=Decimal("104.37"),
            initial_value=Decimal("100"),
            high_watermark=Decimal("104.37"),
            cumulative_log_return=math.log(1.0437),
            cumulative_benchmark_log_return=math.log(1.1673),
        ),
    )
    assert 0.0 <= rb.r_total <= 1.0


def test_randomized_bounds_invariant() -> None:
    rng = random.Random(0)
    for _ in range(200):
        cum_log = rng.uniform(-2.0, 2.0)
        bench = rng.uniform(-1.5, 1.5)
        v_after = max(0.01, 100.0 * math.exp(cum_log))
        rb = compute_composite_reward(
            **_kw(
                value_after=v_after,
                cumulative_log_return=cum_log,
                cumulative_benchmark_log_return=bench,
                turnover_ratio=rng.uniform(0.0, 5.0),
                hhi=rng.uniform(0.0, 1.0),
                gross_leverage=rng.uniform(0.0, 0.99),
                high_watermark=max(v_after, 100.0),
            ),
        )
        for name in (
            "c_alpha",
            "c_return",
            "c_drawdown",
            "c_solvency",
            "c_efficiency",
            "c_diversity",
            "c_consistency",
            "g_compliance",
            "r_total",
        ):
            value = getattr(rb, name)
            assert 0.0 <= value <= 1.0, (name, value)


def test_broken_baseline_replay_score_band() -> None:
    rb = compute_composite_reward(
        **_kw(
            value_after=104.37,
            initial_value=100.0,
            cumulative_log_return=math.log(1.0437),
            cumulative_benchmark_log_return=math.log(1.1673),
            high_watermark=104.37,
            turnover_ratio=0.05,
            hhi=0.20,
        ),
    )
    assert 0.50 <= rb.r_total <= 0.65


def test_strong_alpha_score_band() -> None:
    rb = compute_composite_reward(
        **_kw(
            value_after=150.0,
            initial_value=100.0,
            cumulative_log_return=math.log(1.50),
            cumulative_benchmark_log_return=math.log(1.1673),
            high_watermark=150.0,
            turnover_ratio=0.10,
            hhi=0.15,
        ),
    )
    assert rb.r_total >= 0.80


def test_loss_score_below_pivot() -> None:
    rb = compute_composite_reward(
        **_kw(
            value_after=90.0,
            initial_value=100.0,
            cumulative_log_return=math.log(0.90),
            cumulative_benchmark_log_return=math.log(1.1673),
            high_watermark=100.0,
            turnover_ratio=0.20,
            hhi=0.30,
        ),
    )
    assert rb.r_total <= 0.50


def test_alpha_K_calibration() -> None:
    rb = compute_composite_reward(**_kw(cumulative_log_return=0.20))
    expected = 1.0 / (1.0 + math.exp(-K_ALPHA * 0.20))
    assert math.isclose(rb.c_alpha, expected, abs_tol=1e-9)


def test_return_K_calibration() -> None:
    rb = compute_composite_reward(**_kw(cumulative_log_return=0.10))
    expected = 1.0 / (1.0 + math.exp(-K_RETURN * 0.10))
    assert math.isclose(rb.c_return, expected, abs_tol=1e-9)
