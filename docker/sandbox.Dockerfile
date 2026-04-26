# TradeBench offline agent sandbox. Build from repository root, for example:
#   docker build -f docker/sandbox.Dockerfile -t tradebench-sandbox:local .
FROM python:3.12.8-slim-bookworm

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY docker/sandbox.requirements.txt /tmp/sandbox.requirements.txt
RUN pip install --no-cache-dir -r /tmp/sandbox.requirements.txt \
    && rm /tmp/sandbox.requirements.txt

RUN groupadd --gid 1000 sandbox \
    && useradd --uid 1000 --gid 1000 -m -s /bin/bash sandbox

USER sandbox
WORKDIR /home/sandbox
