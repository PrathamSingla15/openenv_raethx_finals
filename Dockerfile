# Multi-stage build for the TradeBench OpenEnv environment.
ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

ARG BUILD_MODE=in-repo
ARG ENV_NAME=tradebench

# Build context is the env root, so copy everything into /app/env.
COPY . /app/env

WORKDIR /app/env

RUN if ! command -v uv >/dev/null 2>&1; then \
        curl -LsSf https://astral.sh/uv/install.sh | sh && \
        mv /root/.local/bin/uv /usr/local/bin/uv && \
        mv /root/.local/bin/uvx /usr/local/bin/uvx; \
    fi

RUN if [ -f uv.lock ]; then \
        uv sync --frozen --no-install-project --no-editable; \
    else \
        uv sync --no-install-project --no-editable; \
    fi

RUN if [ -f uv.lock ]; then \
        uv sync --frozen --no-editable; \
    else \
        uv sync --no-editable; \
    fi

# Final runtime stage
FROM ${BASE_IMAGE}

WORKDIR /app

COPY --from=builder /app/env/.venv /app/.venv
COPY --from=builder /app/env /app/env
# uv-managed Python interpreter the venv's symlinks resolve to.
COPY --from=builder /root/.local/share/uv /root/.local/share/uv
# Web UI loads README from /app/README.md
COPY --from=builder /app/env/README.md /app/README.md

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/env:$PYTHONPATH"
# Pin the dataset root explicitly. The installed tradebench wheel's
# `_project_root()` resolves to /app/.venv/lib/python3.12 (wrong: it's
# the venv's parent, not the repo). Setting this env var skips that
# broken resolver in resolve_settings() and ensures the env reads from
# the catalog + real-data shipped under /app/env/datasets/.
ENV TRADEBENCH_DATASET_ROOT="/app/env/datasets"
# Intentionally NOT setting ENABLE_WEB_INTERFACE=true. openenv-core's default
# Gradio UI would compete for the root mount and shadow our custom UI.
# server/app.py mounts ``server.ui.build_ui()`` at / with our charcoal theme
# + NEON_CSS; the env's create_app returns a plain FastAPI app
# (ENABLE_WEB_INTERFACE defaults to false) so our mount is the only one.
# A legacy /web -> / redirect (308) preserves backwards compatibility for
# anyone holding the previous URL.

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["sh", "-c", "cd /app/env && uvicorn server.app:app --host 0.0.0.0 --port 8000"]
