"""Regenerate every submission figure.

Usage:
    uv run python -m scripts.visualize.render_all
"""

from __future__ import annotations

from pathlib import Path

from scripts.visualize import (
    viz_a_reward,
    viz_b_reward_roi,
    viz_d_bar_alpha,
    viz_e_action_mix,
    viz_f_reward_components,
    viz_g_equity_curves_all,
)
from scripts.visualize.data_loader import PROJECT_ROOT

OUT_DIR = PROJECT_ROOT / "docs" / "figures"

QWEN = "qwen-qwen3-32b-groq"
GLM = "zai-glm-5.1-together"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Combined headline reward-evolution chart (both models on one axes).
    viz_a_reward.render(OUT_DIR / "reward_evolution.png")
    print("  OK  reward_evolution.png  (both models)")

    # Per-model plots: Qwen (un-suffixed) + GLM (_glm suffix).
    for model_id, suffix in [(QWEN, ""), (GLM, "_glm")]:
        viz_b_reward_roi.render(OUT_DIR / f"reward_roi_combined{suffix}.png", model_id=model_id)
        print(f"  OK  reward_roi_combined{suffix}.png")

        viz_d_bar_alpha.render(OUT_DIR / f"bar_alpha_vs_bnh{suffix}.png", model_id=model_id)
        print(f"  OK  bar_alpha_vs_bnh{suffix}.png")

        viz_g_equity_curves_all.render(
            OUT_DIR / f"equity_curves_all_iters{suffix}.png", model_id=model_id,
        )
        print(f"  OK  equity_curves_all_iters{suffix}.png")

        viz_e_action_mix.render(OUT_DIR / f"action_mix_evolution{suffix}.png", model_id=model_id)
        print(f"  OK  action_mix_evolution{suffix}.png")

        viz_f_reward_components.render(
            OUT_DIR / f"reward_components_baseline_vs_final{suffix}.png", model_id=model_id,
        )
        print(f"  OK  reward_components_baseline_vs_final{suffix}.png")

    print(f"\nAll figures written to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
