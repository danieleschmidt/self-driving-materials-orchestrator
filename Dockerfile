FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r materials && useradd -r -g materials materials

# Set work directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY pyproject.toml ./
COPY README.md ./
COPY LICENSE ./

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -e . && \
    pip cache purge

# Copy source code
COPY src/ ./src/
COPY docs/ ./docs/

# Create directories for data and logs
RUN mkdir -p /app/data /app/logs && \
    chown -R materials:materials /app

# Switch to non-root user
USER materials

# Expose ports
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Default command
CMD ["python", "-m", "materials_orchestrator.cli", "launch", "--port", "8000"]