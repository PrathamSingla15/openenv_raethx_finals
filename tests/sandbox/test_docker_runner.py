"""Sandbox provider tests (mocked Docker + local workspace semantics)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

from tradebench.sandbox.models import SandboxRunRequest
from tradebench.sandbox.providers.docker import DockerSandboxRunner
from tradebench.sandbox.providers.local import LocalSandboxRunner
from tradebench.sandbox.workspace import materialize_episode_workspace

REPO_ROOT = Path(__file__).resolve().parents[2]
SECCOMP_PROFILE = REPO_ROOT / "docker" / "constraints" / "seccomp.json"


def test_docker_runner_disables_network_and_hardens_container(tmp_path: Path) -> None:
    """Mocked Docker SDK: verify offline-oriented create() configuration."""

    mock_container = MagicMock()
    mock_container.wait.return_value = {"StatusCode": 0}
    mock_container.logs.side_effect = lambda stdout=True, stderr=False: (
        b"ok\n" if stdout else b""
    )

    captured: dict[str, object] = {}

    def _create(*args: object, **kwargs: object) -> MagicMock:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return mock_container

    client = MagicMock()
    client.containers.create = MagicMock(side_effect=_create)

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "sample.txt").write_text("x", encoding="ascii")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="ascii")
    ws = materialize_episode_workspace(
        artifact_root=tmp_path / "artifacts",
        episode_id="mock-ep",
        dataset_host_path=dataset,
        manifest_source_path=manifest,
    )

    runner = DockerSandboxRunner(
        client,
        image="tradebench/sandbox:test",
        seccomp_profile_host_path=SECCOMP_PROFILE,
    )
    req = SandboxRunRequest(
        command=["python", "-c", "print('ok')"],
        episode_workspace=ws,
    )
    result = runner.run(req)

    assert result.exit_code == 0
    assert "ok" in result.stdout
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs.get("network_disabled") is True
    assert kwargs.get("use_config_proxy") is False
    assert kwargs.get("read_only") is True
    assert kwargs.get("user") == "1000:1000"

    sec = kwargs.get("security_opt") or []
    assert "no-new-privileges:true" in sec
    assert any(str(x).startswith("seccomp=") for x in sec)

    tmpfs = kwargs.get("tmpfs") or {}
    assert "/tmp" in tmpfs

    mounts = kwargs.get("mounts") or []
    by_target = {m.get("Target"): m for m in mounts}
    assert by_target["/workspace/data"].get("ReadOnly") is True
    assert by_target["/workspace/meta"].get("ReadOnly") is True
    assert by_target["/workspace/work"].get("ReadOnly") is False
    assert by_target["/workspace/out"].get("ReadOnly") is False

    sources = {str(m.get("Source", "")) for m in mounts}
    assert str(Path.home()) not in sources

    mock_container.start.assert_called_once()
    mock_container.remove.assert_called_once_with(force=True)


def test_episode_workspace_writable_workdir_per_episode(tmp_path: Path) -> None:
    """Local runner: each episode materializes an isolated writable working tree."""

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "fixture.txt").write_text("benchmark", encoding="ascii")

    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"task_id": "t1"}', encoding="ascii")

    ws_a = materialize_episode_workspace(
        artifact_root=tmp_path / "artifacts",
        episode_id="ep-a",
        dataset_host_path=dataset,
        manifest_source_path=manifest,
    )
    ws_b = materialize_episode_workspace(
        artifact_root=tmp_path / "artifacts",
        episode_id="ep-b",
        dataset_host_path=dataset,
        manifest_source_path=manifest,
    )

    runner = LocalSandboxRunner()
    res_a = runner.run(
        SandboxRunRequest(
            command=[
                sys.executable,
                "-c",
                "open('marker.txt', 'w', encoding='utf-8').write('a')",
            ],
            episode_workspace=ws_a,
        )
    )
    res_b = runner.run(
        SandboxRunRequest(
            command=[
                sys.executable,
                "-c",
                "open('marker.txt', 'w', encoding='utf-8').write('b')",
            ],
            episode_workspace=ws_b,
        )
    )

    assert res_a.exit_code == 0
    assert res_b.exit_code == 0
    assert (ws_a.host_work_dir / "marker.txt").read_text(encoding="utf-8") == "a"
    assert (ws_b.host_work_dir / "marker.txt").read_text(encoding="utf-8") == "b"
    path_a = ws_a.host_work_dir / "marker.txt"
    path_b = ws_b.host_work_dir / "marker.txt"
    assert not path_a.samefile(path_b)


def test_local_runner_decodes_utf8_with_replacement(tmp_path: Path) -> None:
    """Local runner should mirror Docker's explicit decode-with-replacement behavior."""

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="ascii")
    ws = materialize_episode_workspace(
        artifact_root=tmp_path / "artifacts",
        episode_id="utf8-ep",
        dataset_host_path=dataset,
        manifest_source_path=manifest,
    )

    runner = LocalSandboxRunner()
    result = runner.run(
        SandboxRunRequest(
            command=[
                sys.executable,
                "-c",
                (
                    "import os, sys; "
                    "os.write(sys.stdout.fileno(), b'bad:\\xff\\n'); "
                    "os.write(sys.stderr.fileno(), b'err:\\xff\\n')"
                ),
            ],
            episode_workspace=ws,
        )
    )

    assert result.exit_code == 0
    assert result.stdout == "bad:\ufffd\n"
    assert result.stderr == "err:\ufffd\n"
