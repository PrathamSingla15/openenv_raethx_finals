"""Kelly-fraction baseline — half-Kelly × 1/N, one-shot, hold-to-end.

Without sandbox access there is no per-asset edge estimate, so the baseline
falls back to a fixed half-Kelly fraction (``0.5``) on the equal-weight
allocation. Compared to ``equal_weight`` (90% invested) this baseline
holds ~50% in cash and ~50% in risky positions, producing:

- Lower terminal wealth in trending bull regimes
- Smaller drawdowns in shock regimes (T2)
- Distinct curve shape on the baseline-vs-trained plot

When a future revision adds a per-asset signal (rolling Sharpe via
``sandbox_exec``), this module is the natural place to compute Kelly
weights ``f_i = μ_i / σ_i²`` and size accordingly.
"""

from __future__ import annotations

from .equal_weight import run_equal_weight_baseline


async def run_kelly_baseline(
    env,
    *,
    tier: str = "t1",
    max_steps: int = 1_000,
    kelly_fraction: float = 0.5,
):
    """Drive ``env`` through one half-Kelly equal-weight episode of ``tier``.

    Returns the same summary shape as :func:`run_equal_weight_baseline` with
    ``name='kelly'`` and the ``kelly_fraction`` baked into ``target_exposure``.
    """

    result = await run_equal_weight_baseline(
        env,
        tier=tier,
        max_steps=max_steps,
        target_exposure=kelly_fraction,
    )
    result["name"] = "kelly"
    return result


__all__ = ["run_kelly_baseline"]
