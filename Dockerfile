# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Dependency builder
#   Uses the official uv image so we get the exact same resolver locally and in CI.
#   We install dependencies into /app/.venv but do NOT install the project itself
#   (--no-install-project) so the layer is fully cacheable: it only rebuilds when
#   pyproject.toml or uv.lock change, not when source files change.
# ─────────────────────────────────────────────────────────────────────────────
FROM ghcr.io/astral-sh/uv:python3.10-bookworm-slim AS builder

WORKDIR /app

# Bring in the lock files first — layer cache stays valid across source changes
COPY pyproject.toml uv.lock ./

# Bake all runtime dependencies into an isolated venv
RUN uv sync --frozen --no-install-project --no-dev


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Runtime image
#   Slim Python base keeps the final image small.
#   System packages are the minimum OpenCV / video processing needs.
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.10-slim AS runtime

WORKDIR /app

# System dependencies required by OpenCV and video processing (ffmpeg for LeRobot)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsm6 \
        libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy the pre-built virtual environment from the builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy application source
COPY augment.py ./
COPY augmentations/ ./augmentations/
COPY utils/ ./utils/

# Make the venv's Python the default interpreter
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# The container acts as a CLI tool — pass flags directly after `docker run`
ENTRYPOINT ["python", "augment.py"]
