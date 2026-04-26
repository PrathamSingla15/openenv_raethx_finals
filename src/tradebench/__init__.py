"""TradeBench: trading benchmark with PIT data and sandboxed agents."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("tradebench")
except PackageNotFoundError:
    __version__ = "0.0.0"
