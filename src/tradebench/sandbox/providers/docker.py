"""Docker-backed offline sandbox runner (non-root, no host secret injection)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

from docker.types import Mount  # type: ignore[import-not-found,import-untyped]

from tradebench.sandbox.models import (
    EpisodeWorkspace,
    SandboxRunRequest,
    SandboxRunResult,
)


class DockerContainerLike(Protocol):
    """Minimal container interface used by ``DockerSandboxRunner``."""

    def start(self) -> None: ...

    def wait(self, **kwargs: int) -> Mapping[str, object]: ...

    def logs(self, *, stdout: bool = True, stderr: bool = False) -> bytes | None: ...

    def remove(self, *, force: bool) -> None: ...


class DockerContainerCollectionLike(Protocol):
    """Subset of ``client.containers`` required by the runner."""

    def create(
        self,
        image: str,
        command: Sequence[str],
        **kwargs: object,
    ) -> DockerContainerLike: ...


class DockerClientLike(Protocol):
    """Minimal Docker client surface accepted by ``DockerSandboxRunner``."""

    @property
    def containers(self) -> DockerContainerCollectionLike: ...


def _workspace_mounts(workspace: EpisodeWorkspace) -> list[Mount]:
    """Return bind mounts with explicit read-only vs read-write modes."""

    ws = workspace
    return [
        Mount(
            target=ws.container_work_dir,
            source=str(ws.host_work_dir.resolve()),
            type="bind",
            read_only=False,
        ),
        Mount(
            target=ws.container_data_dir,
            source=str(ws.host_data_dir.resolve()),
            type="bind",
            read_only=True,
        ),
        Mount(
            target=ws.container_meta_dir,
            source=str(ws.host_meta_dir.resolve()),
            type="bind",
            read_only=True,
        ),
        Mount(
            target=ws.container_output_dir,
            source=str(ws.host_output_dir.resolve()),
            type="bind",
            read_only=False,
        ),
    ]


def _minimal_container_env(
    request_env: dict[str, str],
    container_paths: dict[str, str] | None = None,
) -> dict[str, str]:
    """Environment visible inside the container (no host os.environ merge).

    ``container_paths`` carries the in-container mount points so an agent
    script using ``os.environ['WORKSPACE_DATA']`` resolves correctly
    regardless of provider (matches the ``LocalSandboxRunner`` injection).
    """

    base: dict[str, str] = {
        "PYTHONUNBUFFERED": "1",
        "PYTHONNOUSERSITE": "1",
        "HOME": "/home/sandbox",
    }
    if container_paths:
        base.update(container_paths)
    return {**base, **request_env}


class DockerSandboxRunner:
    """Run agent commands in the sandbox image with hardened Docker flags.

    Intended for benchmark runs:

    - No container networking (``network_disabled``)
    - Read-only container root except workspace bind mounts
    - Non-root ``user`` (must match the image USER / uid)
    - ``no-new-privileges`` and host seccomp profile path
    - ``use_config_proxy=False`` (no Docker client proxy env injection)
    - No implicit host home-directory mount; only episode workspace paths
    """

    def __init__(
        self,
        client: DockerClientLike,
        *,
        image: str,
        seccomp_profile_host_path: Path,
        user: str = "1000:1000",
    ) -> None:
        """Pass a Docker client exposing ``client.containers.create(...)``."""

        self._client = client
        self._image = image
        self._seccomp = seccomp_profile_host_path.expanduser().resolve()
        self._user = user
        if not self._seccomp.is_file():
            msg = f"seccomp profile must exist: {self._seccomp}"
            raise ValueError(msg)

    def run(self, request: SandboxRunRequest) -> SandboxRunResult:
        ws = request.episode_workspace
        mounts = _workspace_mounts(ws)
        security_opt = [
            "no-new-privileges:true",
            f"seccomp={self._seccomp}",
        ]
        environment = _minimal_container_env(
            dict(request.env),
            container_paths={
                "WORKSPACE_DATA": ws.container_data_dir,
                "WORKSPACE_WORK": ws.container_work_dir,
                "WORKSPACE_META": ws.container_meta_dir,
                "WORKSPACE_OUT": ws.container_output_dir,
            },
        )
        container = self._client.containers.create(
            self._image,
            request.command,
            user=self._user,
            working_dir=ws.container_work_dir,
            network_disabled=True,
            use_config_proxy=False,
            environment=environment,
            read_only=True,
            mounts=mounts,
            security_opt=security_opt,
            tmpfs={"/tmp": "rw,nosuid,size=67108864"},
        )
        try:
            container.start()
            wait_kw: dict[str, int] = {}
            if request.timeout_seconds is not None:
                wait_kw["timeout"] = request.timeout_seconds
            result = container.wait(**wait_kw)
            status_code = result.get("StatusCode")
            if not isinstance(status_code, int):
                msg = f"container wait() returned non-int StatusCode: {status_code!r}"
                raise TypeError(msg)
            code = status_code
            out_b = container.logs(stdout=True, stderr=False) or b""
            err_b = container.logs(stdout=False, stderr=True) or b""
        finally:
            container.remove(force=True)
        return SandboxRunResult(
            exit_code=code,
            stdout=out_b.decode(errors="replace"),
            stderr=err_b.decode(errors="replace"),
        )


__all__ = ["DockerSandboxRunner"]
