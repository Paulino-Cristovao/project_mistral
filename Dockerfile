FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
WORKDIR /app
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e .

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy application code
COPY src/ ./src/
COPY examples/ ./examples/
COPY data/ ./data/

# Create directories for outputs
RUN mkdir -p /app/reports /app/audit_logs /app/compliance_reports

# Set permissions
RUN chmod -R 755 /app

# Expose port for Streamlit dashboard
EXPOSE 8501

# Default command
CMD ["python", "-m", "docbridgeguard.cli", "--help"]