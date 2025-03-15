FROM python:3.10-slim as builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends     build-essential     libpq-dev     && apt-get clean     && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY README.md .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip &&     pip install --no-cache-dir -e .

# Second stage, copy only what's needed for production
FROM python:3.10-slim

WORKDIR /app

# Copy virtual environment from the builder
COPY --from=builder /usr/local /usr/local

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends     libpq5     && apt-get clean     && rm -rf /var/lib/apt/lists/*

# Copy application
COPY app app/
COPY alembic.ini .
COPY alembic alembic/

# Create non-root user
RUN useradd -m appuser
USER appuser

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Run gunicorn
CMD ["gunicorn", "app.main:app", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
