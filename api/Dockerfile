FROM python:3.11-slim AS base

WORKDIR /app

# System deps for torch CPU builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Install UV for fast dependency resolution
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Copy dependency manifests first (layer caching)
COPY api/pyproject.toml ./api/

# Install API dependencies
RUN cd api && uv pip install --system -r pyproject.toml

# Copy application code — app/ modules use flat imports,
# so we add /app/app to PYTHONPATH.
COPY app/*.py ./app/
COPY api/ ./api/

# Copy model artifacts (overridable via volume mount)
COPY models/ ./models/

ENV PYTHONPATH="/app/app:${PYTHONPATH}"

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
