FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    pkg-config \
    default-libmysqlclient-dev \
    poppler-utils \
    libreoffice-core \
    libreoffice-writer \
    libreoffice-calc \
    libreoffice-impress \
    && rm -rf /var/lib/apt/lists/*

# Create app user with explicit UID for predictable bind mount permissions
RUN useradd --create-home --shell /bin/bash --uid 1000 appuser

# Set work directory
WORKDIR /app

# Copy requirements first for better caching
COPY pyproject.toml ./
COPY backend/__init__.py backend/
COPY backend/app/__init__.py backend/app/

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -e .

# Copy application code
COPY backend/ backend/
COPY scripts/ scripts/
COPY alembic.ini ./

# Create directories
RUN mkdir -p /data/artifacts /data/chat_files /var/log/mindrouter && \
    chown -R appuser:appuser /app /data /var/log/mindrouter

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

# Run the application
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--timeout-graceful-shutdown", "60"]
