FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# Copy project files
COPY pyproject.toml uv.lock ./
COPY src/ ./src/
COPY config/ ./config/
COPY tests/ ./tests/

# Install dependencies
RUN uv sync --frozen --no-dev

# Create data directories
RUN mkdir -p /data/chroma /data/docs

# Expose port
EXPOSE 8000

# Default command
CMD ["uv", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]