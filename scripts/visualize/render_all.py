"""Regenerate every submission figure.

Usage:
    uv run python -m scripts.visualize.render_all
"""

from __future__ import annotations

from pathlib import Path

from scripts.visualize import (
    viz_a_reward,
    viz_b_reward_roi,
    viz_c_prompts,
    viz_d_bar_alpha,
    viz_e_action_mix,
    viz_f_reward_components,
)
from scripts.visualize.data_loader import PROJECT_ROOT

OUT_DIR = PROJECT_ROOT / "docs" / "figures"

_TARGETS = [
    (viz_a_reward.render, "reward_evolution.png"),
    (viz_b_reward_roi.render, "reward_roi_combined.png"),
    (viz_c_prompts.render, "prompt_evolution.png"),
    (viz_d_bar_alpha.render, "bar_alpha_vs_bnh.png"),
    (viz_e_action_mix.render, "action_mix_evolution.png"),
    (viz_f_reward_components.render, "reward_components_baseline_vs_final.png"),
]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for render_fn, fname in _TARGETS:
        out = OUT_DIR / fname
        try:
            render_fn(out)
            print(f"  OK  {fname}")
        except Exception as e:
            print(f"  FAIL {fname}: {e!r}")
            raise
    print(f"\nAll figures written to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
