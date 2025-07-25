# ----------- Builder Stage -----------
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04 AS builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3.11-dev python3-pip \
    build-essential git curl && \
    rm -rf /var/lib/apt/lists/*

# Set up virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python3.11 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install pip and cython first to avoid build issues
RUN pip install --upgrade pip cython

# Install Python dependencies
COPY requirements.txt requirements.txt
COPY requirements-dev.txt requirements-dev.txt
RUN pip install --no-cache-dir torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128 && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-dev.txt

# ----------- Runtime Stage -----------
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04 AS runtime

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3.11-dev curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set workdir and copy project files
WORKDIR /app
COPY . /app

# Expose FastAPI and MLflow ports
EXPOSE 8000 5000

# Healthcheck for FastAPI
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Default command: start API (can be overridden)
CMD ["python", "scripts/run_api.py"] 