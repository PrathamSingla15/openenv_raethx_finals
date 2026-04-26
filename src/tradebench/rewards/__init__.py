"""Multi-signal composite reward and anti-hack safeguards.

Public API:

- ``RewardBreakdown`` — dataclass of per-component reward terms emitted by
  ``advance_day``, embedded in ``TradeObservation.reward_breakdown``.
- ``compute_composite_reward`` — pure function mapping portfolio / behavior
  signals into a ``RewardBreakdown`` with bounded secondary terms.
- ``scan_forbidden_globals`` — regex scanner over sandbox stdout/stderr or
  source for dangerous Python constructs.
- ``check_rules_clause`` — regex scanner over agent text output for phrases
  that indicate the agent is recalling specific historical events.
"""

from __future__ import annotations

from .anti_hack import check_rules_clause, scan_forbidden_globals
from .composite import RewardBreakdown, compute_composite_reward

__all__ = [
    "RewardBreakdown",
    "check_rules_clause",
    "compute_composite_reward",
    "scan_forbidden_globals",
]
