"""In-process sandbox runner for fast tests (no isolation, not a security boundary)."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Sequence

from tradebench.sandbox.models import SandboxRunRequest, SandboxRunResult


class LocalSandboxRunner:
    """Execute commands directly on the host for unit tests.

    This runner intentionally does **not** provide confidentiality, integrity, or
    availability guarantees. It shares the host filesystem and environment
    beyond the workspace working directory and must never be advertised as a
    substitute for the Docker-backed sandbox used in real benchmark runs.

    Container-vs-host path bridging:
        The Docker provider mounts the workspace inside the container at
        ``/workspace/{data,work,meta,out}``. The local provider has no
        such mount — running ``open('/workspace/data/...')`` on the host
        fails. To keep agent scripts portable, this runner injects the
        following environment variables so scripts can resolve the right
        path regardless of provider:

        - ``WORKSPACE_DATA``  → host_data_dir
        - ``WORKSPACE_WORK``  → host_work_dir
        - ``WORKSPACE_META``  → host_meta_dir
        - ``WORKSPACE_OUT``   → host_output_dir

        Inside Docker these env vars are also exported (matching the
        container mount points), so an agent script using
        ``os.environ['WORKSPACE_DATA']`` is provider-agnostic.
    """

    def __init__(self, inherit_host_env: bool = True) -> None:
        self._inherit_host_env = inherit_host_env

    def run(self, request: SandboxRunRequest) -> SandboxRunResult:
        cmd: Sequence[str] = request.command
        ws = request.episode_workspace
        workspace_env = {
            "WORKSPACE_DATA": str(ws.host_data_dir),
            "WORKSPACE_WORK": str(ws.host_work_dir),
            "WORKSPACE_META": str(ws.host_meta_dir),
            "WORKSPACE_OUT": str(ws.host_output_dir),
        }
        env: dict[str, str]
        if self._inherit_host_env:
            env = {**os.environ, **workspace_env, **request.env}
        else:
            env = {**workspace_env, **request.env}
        proc = subprocess.run(
            list(cmd),
            cwd=ws.host_work_dir,
            env=env,
            capture_output=True,
            timeout=request.timeout_seconds,
            check=False,
        )
        return SandboxRunResult(
            exit_code=int(proc.returncode),
            stdout=(proc.stdout or b"").decode("utf-8", errors="replace"),
            stderr=(proc.stderr or b"").decode("utf-8", errors="replace"),
        )


__all__ = ["LocalSandboxRunner"]
