# Multi-stage Dockerfile for Price Predictor Application
# Optimized for production deployment with security best practices

# ===============================
# Build Stage
# ===============================
FROM python:3.9-slim as builder

# Set build arguments
ARG BUILD_DATE
ARG VERSION
ARG VCS_REF

# Add metadata labels
LABEL maintainer="1998prakhargupta <1998prakhargupta@gmail.com>" \
      org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="price-predictor" \
      org.label-schema.description="Stock Price Prediction System" \
      org.label-schema.version=$VERSION \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.vcs-url="https://github.com/1998prakhargupta/price-predictor" \
      org.label-schema.schema-version="1.0"

# Set environment variables for build
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create application directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --user --no-cache-dir -r requirements.txt

# ===============================
# Production Stage  
# ===============================
FROM python:3.9-slim as production

# Set production environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    APP_ENV=production \
    LOG_LEVEL=INFO

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r appuser && \
    useradd -r -g appuser -d /app -s /bin/bash appuser

# Set working directory
WORKDIR /app

# Copy Python packages from builder stage
COPY --from=builder /root/.local /home/appuser/.local

# Make sure scripts in .local are usable:
ENV PATH=/home/appuser/.local/bin:$PATH

# Copy application code
COPY --chown=appuser:appuser . .

# Create necessary directories with proper permissions
RUN mkdir -p /app/data/outputs /app/data/cache /app/data/logs /app/data/reports && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Expose port
EXPOSE 8000

# Default command
CMD ["python", "-m", "src.main"]

# ===============================
# Development Stage
# ===============================
FROM production as development

# Switch back to root to install dev dependencies
USER root

# Install development dependencies
RUN pip install --no-cache-dir -r requirements-dev.txt

# Install additional development tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Copy development configuration
COPY --chown=appuser:appuser .env.example .env

# Switch back to appuser
USER appuser

# Override command for development
CMD ["python", "-m", "src.main", "--debug"]

# ===============================
# Testing Stage
# ===============================
FROM development as testing

# Switch to root for test setup
USER root

# Install test-specific tools
RUN pip install --no-cache-dir pytest-xdist pytest-cov

# Copy test configuration
COPY --chown=appuser:appuser pytest.ini tox.ini ./

# Switch back to appuser
USER appuser

# Run tests by default
CMD ["pytest", "tests/", "-v", "--cov=src"]

# ===============================
# Jupyter/Notebook Stage
# ===============================
FROM development as jupyter

# Switch to root for jupyter installation
USER root

# Install Jupyter and extensions
RUN pip install --no-cache-dir \
    jupyter \
    jupyterlab \
    ipywidgets \
    jupyter-contrib-nbextensions

# Install Jupyter extensions
RUN jupyter contrib nbextension install --system && \
    jupyter nbextension enable --py widgetsnbextension --sys-prefix

# Create jupyter config directory
RUN mkdir -p /home/appuser/.jupyter && \
    chown -R appuser:appuser /home/appuser/.jupyter

# Switch back to appuser
USER appuser

# Expose Jupyter port
EXPOSE 8888

# Start Jupyter
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]

# ===============================
# Production-Optimized Stage
# ===============================
FROM python:3.9-alpine as production-alpine

# Install runtime dependencies
RUN apk add --no-cache \
    curl \
    ca-certificates \
    && addgroup -g 1000 appuser \
    && adduser -D -u 1000 -G appuser appuser

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    APP_ENV=production

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY --chown=appuser:appuser . .

# Create data directories
RUN mkdir -p data/outputs data/cache data/logs data/reports && \
    chown -R appuser:appuser data/

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

CMD ["python", "-m", "src.main"]

# ===============================
# GPU-Enabled Stage (for ML training)
# ===============================
FROM nvidia/cuda:11.8-runtime-ubuntu20.04 as gpu

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-pip \
    python3.9-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -s /usr/bin/python3.9 /usr/bin/python

# Create user
RUN groupadd -r appuser && \
    useradd -r -g appuser -d /app appuser

WORKDIR /app

# Install PyTorch with CUDA support
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY --chown=appuser:appuser . .

USER appuser

CMD ["python", "-m", "src.models.train", "--gpu"]

# ===============================
# Microservice Stage (API only)
# ===============================
FROM python:3.9-slim as api

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_ENV=production

# Install minimal dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create user
RUN groupadd -r apiuser && \
    useradd -r -g apiuser -d /app apiuser

WORKDIR /app

# Install only API dependencies
COPY requirements-api.txt* ./
RUN pip install --no-cache-dir fastapi uvicorn

# Copy only API code
COPY --chown=apiuser:apiuser src/api/ src/api/
COPY --chown=apiuser:apiuser src/models/ src/models/
COPY --chown=apiuser:apiuser src/utils/ src/utils/

USER apiuser

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ===============================
# Build Arguments and Metadata
# ===============================

# Build arguments for versioning
ARG GIT_COMMIT
ARG BUILD_NUMBER
ARG BUILD_BRANCH

# Additional labels
LABEL org.opencontainers.image.title="Price Predictor" \
      org.opencontainers.image.description="ML-based stock price prediction system" \
      org.opencontainers.image.vendor="Price Predictor Team" \
      org.opencontainers.image.version=$VERSION \
      org.opencontainers.image.revision=$GIT_COMMIT \
      org.opencontainers.image.source="https://github.com/1998prakhargupta/price-predictor" \
      org.opencontainers.image.documentation="https://github.com/1998prakhargupta/price-predictor/docs" \
      build.number=$BUILD_NUMBER \
      build.branch=$BUILD_BRANCH
