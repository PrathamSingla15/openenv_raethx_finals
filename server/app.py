"""FastAPI application for the TradeBench Environment.

Endpoints (provided by ``openenv.core.env_server.http_server.create_app``):
    - POST /reset   Reset the environment (accepts ``task_tier``, ``seed``, ``episode_id``)
    - POST /step    Execute a TradeAction
    - GET  /state   Current TradeState snapshot
    - GET  /schema  Action / Observation / State JSON schemas
    - GET  /health  Liveness probe
    - WS   /ws      Persistent session for multi-step episodes
"""

from __future__ import annotations

import logging

try:
    from openenv.core.env_server.http_server import create_app
except Exception as exc:  # pragma: no cover - import-time guard
    raise ImportError(
        "openenv-core is required. Install with: pip install openenv-core"
    ) from exc

try:
    from ..models import TradeAction, TradeObservation
    from .tradebench_environment import TradeBenchEnvironment
except (ImportError, ModuleNotFoundError):  # pragma: no cover - dev import fallback
    from models import TradeAction, TradeObservation  # type: ignore[no-redef]
    from server.tradebench_environment import TradeBenchEnvironment  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

app = create_app(
    TradeBenchEnvironment,
    TradeAction,
    TradeObservation,
    env_name="tradebench",
    max_concurrent_envs=10,
)


# Backwards compatibility: the previous deployment served the UI at /web.
# Register this BEFORE the Gradio mount at / so the explicit handler wins
# over the catch-all mount.
@app.get("/web")
@app.get("/web/")
async def _web_legacy_redirect():
    from fastapi.responses import RedirectResponse

    return RedirectResponse(url="/", status_code=308)


# Lazy gradio import keeps the FastAPI surface up even if gradio fails to load
# (partial install, container troubleshooting).
#
# The Gradio UI mounts at ``/`` rather than ``/web`` because the HF Spaces
# reverse proxy strips trailing slashes from Location headers, which turns the
# FastAPI ``/web`` -> ``/web/`` slash redirect into a 307 loop (the symptom
# observed pre-fix: every ``GET /web`` logged a 307 and the browser never
# resolved). Mounting at root sidesteps the slash-redirect entirely. OpenEnv
# routes (/reset, /step, /state, /schema, /health, /ws, /docs, /openapi.json)
# are registered before the mount so they keep precedence over the Gradio
# catch-all.
try:
    import gradio as gr

    from .ui import DARK_THEME, NEON_CSS, build_ui

    _gradio_app = build_ui()
    # Gradio 6 reads ``theme`` / ``css`` at mount time, not at Blocks construction.
    _mount_kwargs: dict = {"path": "/"}
    import inspect as _inspect
    _mount_sig = _inspect.signature(gr.mount_gradio_app).parameters
    if "theme" in _mount_sig:
        _mount_kwargs["theme"] = DARK_THEME
    if "css" in _mount_sig:
        _mount_kwargs["css"] = NEON_CSS
    app = gr.mount_gradio_app(app, _gradio_app, **_mount_kwargs)
    _gradio_mounted = True
    logger.info("Mounted Gradio UI at /")
except Exception as exc:  # pragma: no cover - mount failures should not crash the API
    _gradio_mounted = False
    logger.warning("Gradio web UI not mounted: %s", exc)


# Fallback root handler when Gradio failed to mount: send the user to the
# OpenAPI docs so the API surface is still discoverable.
if not _gradio_mounted:

    @app.get("/")
    async def _root_fallback():
        from fastapi.responses import RedirectResponse

        return RedirectResponse(url="/docs")


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Entry point for direct execution."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
