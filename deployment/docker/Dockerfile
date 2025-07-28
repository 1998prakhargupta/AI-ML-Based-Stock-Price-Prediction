# =============================================================================
# Enterprise Stock Price Predictor - Unified Multi-Stage Dockerfile
# =============================================================================
# Production-ready containerization with multiple deployment targets
# Supports development, production, ML training, inference, and data pipeline

# ================================
# Build Arguments and Metadata
# ================================
ARG PYTHON_VERSION=3.10
ARG DEBIAN_VERSION=slim
ARG BUILD_DATE
ARG VERSION=1.0.0
ARG VCS_REF
ARG MAINTAINER="1998prakhargupta <1998prakhargupta@gmail.com>"

# ================================
# Stage 1: Base Environment Setup
# ================================
FROM python:${PYTHON_VERSION}-${DEBIAN_VERSION} as base

# Add metadata labels
LABEL maintainer="${MAINTAINER}" \
      org.opencontainers.image.title="Price Predictor" \
      org.opencontainers.image.description="ML-based stock price prediction system" \
      org.opencontainers.image.vendor="Price Predictor Team" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.source="https://github.com/1998prakhargupta/price-predictor" \
      org.opencontainers.image.documentation="https://github.com/1998prakhargupta/price-predictor/docs"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    APP_ENV=production \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libpq-dev \
    libffi-dev \
    libssl-dev \
    gcc \
    g++ \
    libc6-dev \
    pkg-config \
    # Graphics and plotting dependencies
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    libfontconfig1 \
    libice6 \
    # Additional ML dependencies
    libblas3 \
    liblapack3 \
    libatlas-base-dev \
    gfortran \
    # Utilities
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create application user for security
RUN groupadd -r appuser \
    && useradd -r -g appuser -d /app -s /bin/bash appuser \
    && mkdir -p /app \
    && chown -R appuser:appuser /app

# Set working directory
WORKDIR /app

# Copy requirements files
COPY requirements.txt requirements-dev.txt ./

# ================================
# Stage 2: Dependencies Installation
# ================================
FROM base as dependencies

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories with proper permissions
RUN mkdir -p \
    data/raw \
    data/processed \
    data/cache \
    data/outputs \
    data/plots \
    data/reports \
    models/experiments \
    models/checkpoints \
    models/production \
    logs \
    config \
    && chown -R appuser:appuser /app

# ================================
# Stage 3: Development Image
# ================================
FROM dependencies as development

# Install development dependencies
RUN pip install --no-cache-dir -r requirements-dev.txt

# Install additional development tools
RUN pip install --no-cache-dir \
    jupyter \
    jupyterlab \
    ipykernel \
    debugpy \
    pre-commit \
    black \
    flake8 \
    pytest \
    pytest-cov \
    pytest-xdist

# Copy application code
COPY --chown=appuser:appuser . .

# Install package in development mode
RUN pip install -e .

# Switch to non-root user
USER appuser

# Expose ports for development
EXPOSE 8000 8888 5678

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Development command with hot reload
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# ================================
# Stage 4: Production Base
# ================================
FROM dependencies as production-base

# Copy only necessary application files
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser config/ ./config/
COPY --chown=appuser:appuser scripts/ ./scripts/
COPY --chown=appuser:appuser run_enterprise_ensemble.py ./
COPY --chown=appuser:appuser setup.py ./
COPY --chown=appuser:appuser README.md ./
COPY --chown=appuser:appuser requirements.txt ./

# Install package
RUN pip install .

# Security: Remove build dependencies and clean up
RUN apt-get remove -y \
    build-essential \
    gcc \
    g++ \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Switch to non-root user
USER appuser

# ================================
# Stage 5: Production API
# ================================
FROM production-base as production

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production command
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

# ================================
# Stage 6: ML Training Image
# ================================
FROM production-base as ml-training

# Switch to root for additional installations
USER root

# Install additional ML dependencies
RUN pip install --no-cache-dir \
    xgboost \
    lightgbm \
    tensorflow \
    torch \
    transformers \
    prophet \
    scikit-optimize \
    optuna \
    mlflow \
    wandb \
    plotly \
    seaborn

# Install CUDA support (optional)
# RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Copy training configurations
COPY --chown=appuser:appuser config/ensemble-config.yaml ./config/
COPY --chown=appuser:appuser config/train-config.yaml ./config/

# Create ML-specific directories
RUN mkdir -p \
    experiments \
    checkpoints \
    tensorboard_logs \
    && chown -R appuser:appuser /app

# Switch back to non-root user
USER appuser

# Expose TensorBoard port
EXPOSE 6006

# Training command
CMD ["python", "run_enterprise_ensemble.py", "--demo"]

# ================================
# Stage 7: Inference Image
# ================================
FROM production-base as inference

# Copy trained models placeholder (models would be mounted in production)
RUN mkdir -p models/production

# Inference-optimized command with fewer workers
CMD ["python", "-m", "uvicorn", "src.api.inference:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]

# ================================
# Stage 8: Data Pipeline Image
# ================================
FROM production-base as data-pipeline

# Switch to root for additional installations
USER root

# Install data processing dependencies
RUN pip install --no-cache-dir \
    apache-airflow \
    celery \
    sqlalchemy \
    psycopg2-binary \
    schedule

# Copy data pipeline specific files
COPY --chown=appuser:appuser scripts/data_pipeline.py ./scripts/
COPY --chown=appuser:appuser config/api-config.yaml ./config/

# Create data directories
RUN mkdir -p \
    data/staging \
    data/validation \
    logs/pipeline \
    && chown -R appuser:appuser /app

# Switch back to non-root user
USER appuser

# Data pipeline command
CMD ["python", "scripts/data_pipeline.py"]

# ================================
# Stage 9: Testing Image
# ================================
FROM development as testing

# Install test-specific tools
RUN pip install --no-cache-dir \
    pytest-xdist \
    pytest-cov \
    pytest-mock \
    pytest-asyncio \
    coverage \
    tox

# Copy test configuration
COPY --chown=appuser:appuser pytest.ini ./
COPY --chown=appuser:appuser tox.ini ./
COPY --chown=appuser:appuser tests/ ./tests/

# Create test reports directory
RUN mkdir -p test-reports

# Run tests by default
CMD ["pytest", "tests/", "-v", "--cov=src", "--cov-report=html", "--cov-report=xml"]

# ================================
# Stage 10: Jupyter/Notebook Image
# ================================
FROM development as jupyter

# Install additional Jupyter extensions
RUN pip install --no-cache-dir \
    jupyterlab-git \
    ipywidgets \
    matplotlib \
    plotly \
    nbconvert

# Copy notebooks
COPY --chown=appuser:appuser notebooks/ ./notebooks/

# Jupyter configuration
RUN mkdir -p /app/.jupyter

# Expose Jupyter port
EXPOSE 8888

# Jupyter command
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]

# ================================
# Stage 11: Monitoring Image
# ================================
FROM production-base as monitoring

# Switch to root for monitoring tools
USER root

# Install monitoring dependencies
RUN pip install --no-cache-dir \
    prometheus-client \
    grafana-api \
    psutil \
    py-cpuinfo

# Copy monitoring scripts
COPY --chown=appuser:appuser scripts/monitoring/ ./scripts/monitoring/
COPY --chown=appuser:appuser config/monitoring/ ./config/monitoring/

# Switch back to non-root user
USER appuser

# Monitoring command
CMD ["python", "scripts/monitoring/metrics_collector.py"]

# ================================
# Build Information
# ================================

# Final build arguments
ARG GIT_COMMIT
ARG BUILD_NUMBER
ARG BUILD_BRANCH

# Additional build metadata
LABEL build.number="${BUILD_NUMBER}" \
      build.branch="${BUILD_BRANCH}" \
      build.commit="${GIT_COMMIT}"
