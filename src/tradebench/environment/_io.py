"""Internal tool I/O dataclasses.

The OpenEnv server (``server/tradebench_environment.py``) consumes these
internally; only the typed Pydantic models in ``models.py`` cross the wire.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

JSONValue = object
JSONObject = dict[str, Any]


@dataclass
class TextBlock:
    """A single human-readable block returned by a tool."""

    text: str
    detail: dict[str, Any] | None = None


@dataclass
class ToolOutput:
    """Tool invocation result.

    ``blocks`` carry the human-facing text. ``metadata`` is the structured
    payload the OpenEnv server surfaces in ``TradeObservation.tool_metadata``.
    ``reward`` and ``finished`` map directly to the OpenEnv StepResult.
    """

    blocks: list[TextBlock] = field(default_factory=list)
    metadata: dict[str, Any] | None = None
    reward: float | None = None
    finished: bool = False


__all__ = ["JSONObject", "JSONValue", "TextBlock", "ToolOutput"]
