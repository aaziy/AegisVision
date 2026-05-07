# syntax=docker/dockerfile:1.7
#
# AegisVision — portable Linux/ARM64 deployment.
#
# Same source code as the Mac native path; only the inference backend
# changes (ONNX Runtime + CPU EP). The image expects the sample video
# bind-mounted at /app/data and writes annotated MP4 + JSONL events
# to /app/output (also a bind mount).
FROM --platform=linux/arm64 ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# OpenCV runtime libs that python:slim doesn't ship.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Layer 1: install deps. Copy the minimum needed to resolve + build the venv,
# then sync. Source goes in next so dep-only changes hit a cached layer.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen

# Layer 2: scripts, configs, ONNX models. (.pt weights are excluded by
# .dockerignore — torch baseline isn't part of the container's job.)
COPY scripts ./scripts
COPY configs ./configs
COPY models/*.onnx ./models/

# Bind-mount targets at runtime; create them so the layer exists.
RUN mkdir -p /app/output /app/data

# Activate the synced venv so we can invoke `python` directly.
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"

ENTRYPOINT ["python", "-m", "aegisvision.pipeline"]
CMD ["--measure", \
     "--backend", "onnx", "--provider", "cpu", \
     "--source", "/app/data/samples/traffic_sample.mp4", \
     "--output-video", "/app/output/annotated.mp4", \
     "--log-file", "/app/output/events.jsonl"]
