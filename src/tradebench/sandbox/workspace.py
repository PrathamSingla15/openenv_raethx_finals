"""Materialize per-episode sandbox directories on the host."""

from __future__ import annotations

import shutil
from pathlib import Path

from tradebench.sandbox.models import EpisodeWorkspace


def materialize_episode_workspace(
    *,
    artifact_root: Path,
    episode_id: str,
    dataset_host_path: Path,
    manifest_source_path: Path | None = None,
) -> EpisodeWorkspace:
    """Create host directories for one episode and optional manifest metadata.

    Layout under ``artifact_root / "sandboxes" / episode_id``:

    - ``work/`` writable agent working directory
    - ``output/`` writable artifacts and logs
    - ``meta/`` read-only *source* for the container (manifest copy lives here)
    - ``dataset_host_path`` is mounted read-only as benchmark data (not copied)

    Args:
        artifact_root: Root under which derived episode files are materialized.
        episode_id: Stable id for this workspace (used as a subdirectory name).
        dataset_host_path: Existing directory to expose as read-only benchmark data.
        manifest_source_path: Optional file copied to ``meta/manifest.json``.
    """

    root = (artifact_root.expanduser().resolve() / "sandboxes" / episode_id).resolve()
    work = root / "work"
    out = root / "output"
    meta = root / "meta"
    data = dataset_host_path.expanduser().resolve()

    if not data.is_dir():
        msg = f"dataset_host_path must be an existing directory: {data}"
        raise ValueError(msg)

    work.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    meta.mkdir(parents=True, exist_ok=True)

    if manifest_source_path is not None:
        src = manifest_source_path.expanduser().resolve()
        if not src.is_file():
            msg = f"manifest_source_path must be an existing file: {src}"
            raise ValueError(msg)
        shutil.copy2(src, meta / "manifest.json")

    return EpisodeWorkspace(
        host_root=root,
        host_work_dir=work,
        host_data_dir=data,
        host_meta_dir=meta,
        host_output_dir=out,
    )
