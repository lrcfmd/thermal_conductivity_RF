# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.12

# Stage 1: build the locked environment with uv
FROM python:${PYTHON_VERSION}-slim AS builder
RUN pip install --no-cache-dir uv
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/
# Install the exact locked dependency set from uv.lock plus the project into
# /app/.venv. --frozen guarantees the runtime stack matches what was tested.
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-editable

# Stage 2: minimal runtime image (same base, so native libs match)
FROM python:${PYTHON_VERSION}-slim
RUN groupadd -r lmds && useradd -r -g lmds -u 1000 -m -d /home/lmds lmds
WORKDIR /app
COPY --from=builder --chown=1000:1000 /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# The platform runs the container as uid 1000. The runner downloads runtime
# assets to ./artifacts, and creates /data/input (file inputs) and /data/output
# (results), so /app and all of /data (and /home/lmds) must be writable by it.
RUN mkdir -p /data/input /data/output && chown -R 1000:1000 /app /data /home/lmds
USER 1000:1000
ENTRYPOINT ["model-run"]
