"""Typed runtime settings for TradeBench."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, field_validator

SandboxProvider = Literal["docker", "local"]

DATASET_ROOT_ENV_VAR = "TRADEBENCH_DATASET_ROOT"
ARTIFACT_ROOT_ENV_VAR = "TRADEBENCH_ARTIFACT_ROOT"
SANDBOX_IMAGE_ENV_VAR = "TRADEBENCH_SANDBOX_IMAGE"
SANDBOX_PROVIDER_ENV_VAR = "TRADEBENCH_SANDBOX_PROVIDER"


class TradeBenchSettings(BaseModel):
    """Configuration for dataset paths, artifacts, and sandbox execution."""

    dataset_root: Path
    artifact_root: Path
    sandbox_image: str
    sandbox_provider: SandboxProvider = "docker"
    deterministic_mode: bool = True

    @field_validator("dataset_root")
    @classmethod
    def dataset_root_must_exist(cls, value: Path) -> Path:
        resolved = value.expanduser().resolve()
        if not resolved.is_dir():
            msg = f"dataset_root must be an existing directory: {resolved}"
            raise ValueError(msg)
        return resolved


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_settings(settings: TradeBenchSettings | None = None) -> TradeBenchSettings:
    """Build a ``TradeBenchSettings`` from environment overrides + defaults.

    Used by the OpenEnv server (``server/tradebench_environment.py``) when
    spinning up a new session. Falls back to the bundled sample dataset
    when ``TRADEBENCH_DATASET_ROOT`` is unset.
    """

    if settings is not None:
        return settings

    from tradebench.data.sample_dataset import ensure_sample_dataset

    dataset_override = os.environ.get(DATASET_ROOT_ENV_VAR)
    if dataset_override:
        dataset_root = Path(dataset_override)
    else:
        dataset_root = ensure_sample_dataset(_project_root() / "datasets")

    artifact_root = Path(
        os.environ.get(ARTIFACT_ROOT_ENV_VAR, str(_project_root() / "artifacts")),
    )
    sandbox_image = os.environ.get(SANDBOX_IMAGE_ENV_VAR, "tradebench:latest")
    sandbox_provider = os.environ.get(SANDBOX_PROVIDER_ENV_VAR, "docker")
    return TradeBenchSettings(
        dataset_root=dataset_root,
        artifact_root=artifact_root,
        sandbox_image=sandbox_image,
        sandbox_provider=sandbox_provider,  # type: ignore[arg-type]
        deterministic_mode=True,
    )


__all__ = [
    "ARTIFACT_ROOT_ENV_VAR",
    "DATASET_ROOT_ENV_VAR",
    "SANDBOX_IMAGE_ENV_VAR",
    "SANDBOX_PROVIDER_ENV_VAR",
    "SandboxProvider",
    "TradeBenchSettings",
    "resolve_settings",
]
