# syntax=docker/dockerfile:1

# ---- Build image ----
FROM python:3.12-slim AS base

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential \
 && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV UV_LINK_MODE=copy

# Create app directory
WORKDIR /app

# Copy dependency manifests first for layer caching
COPY pyproject.toml uv.lock ./

# Sync/install deps (prod)
RUN /root/.local/bin/uv sync --frozen --no-dev --python=3.12

# Copy source code
COPY src ./src
COPY tests ./tests
COPY main.py ./
COPY start.py ./

# Runtime env
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8123

# Expose service ports: FastAPI (8123) and MCP (8000)
EXPOSE 8123 8000

# Start both MCP (as background thread) and FastAPI via start.py
CMD ["/root/.local/bin/uv", "run", "python", "start.py"]
