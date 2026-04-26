"""Pydantic models for sandbox workspaces and run requests."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field


class EpisodeWorkspace(BaseModel):
    """Host paths for a single episode sandbox and fixed in-container mount targets.

    Mount modes inside the container (enforced by the Docker provider):

    - ``/workspace/work``: read-write bind of ``host_work_dir``
    - ``/workspace/data``: read-only bind of ``host_data_dir`` (benchmark tables)
    - ``/workspace/meta``: read-only bind of ``host_meta_dir`` (plan / manifest copies)
    - ``/workspace/out``: read-write bind of ``host_output_dir`` (artifacts, logs)
    """

    model_config = ConfigDict(frozen=True)

    host_root: Path
    host_work_dir: Path
    host_data_dir: Path
    host_meta_dir: Path
    host_output_dir: Path

    container_work_dir: str = "/workspace/work"
    container_data_dir: str = "/workspace/data"
    container_meta_dir: str = "/workspace/meta"
    container_output_dir: str = "/workspace/out"


class SandboxRunRequest(BaseModel):
    """A command to execute inside the episode workspace."""

    model_config = ConfigDict(frozen=True)

    command: list[str]
    episode_workspace: EpisodeWorkspace
    env: dict[str, str] = Field(default_factory=dict)
    timeout_seconds: int | None = None


class SandboxRunResult(BaseModel):
    """Process outcome from a sandbox run."""

    model_config = ConfigDict(frozen=True)

    exit_code: int
    stdout: str
    stderr: str


class SandboxRunner(Protocol):
    """Minimal runner surface shared by Docker and local providers."""

    def run(self, request: SandboxRunRequest) -> SandboxRunResult:
        """Execute ``request.command`` with workspace-mounted semantics."""
        ...
