"""Top-level re-exports for the TradeBench OpenEnv shell.

The relative imports below assume this file is being loaded as the package
init for an installed ``tradebench`` wheel (where hatchling places it as a
top-level module). When pytest discovers it during in-tree collection the
repo root isn't a real package, so the relative form fails — we fall back
to absolute imports against ``pythonpath = ["src"]`` plus the rootdir.
"""

try:
    from .client import TradeBenchEnv  # type: ignore[no-redef]
    from .models import (  # type: ignore[no-redef]
        TradeAction,
        TradeObservation,
        TradeState,
    )
except ImportError:  # pragma: no cover - in-tree dev / pytest collection
    from client import TradeBenchEnv  # type: ignore[no-redef]
    from models import (  # type: ignore[no-redef]
        TradeAction,
        TradeObservation,
        TradeState,
    )

__all__ = [
    "TradeBenchEnv",
    "TradeAction",
    "TradeObservation",
    "TradeState",
]
