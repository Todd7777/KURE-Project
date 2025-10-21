# Multi-stage build for Nonlinear Rectified Flows
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    curl \
    vim \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Development stage
FROM base AS development

WORKDIR /workspace

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install -r requirements.txt

# Install development dependencies
RUN pip install \
    pytest>=7.4.0 \
    pytest-cov>=4.1.0 \
    pytest-xdist>=3.5.0 \
    black>=23.0.0 \
    isort>=5.12.0 \
    flake8>=6.1.0 \
    mypy>=1.5.0 \
    pre-commit>=3.5.0 \
    ipython \
    jupyter

# Copy project files
COPY . .

# Install package in editable mode
RUN pip install -e .

# Set up pre-commit hooks
RUN git init . || true && pre-commit install || true

# Production stage
FROM base AS production

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy only necessary files
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY setup.py pyproject.toml README.md ./

# Install package
RUN pip install .

# Create directories for data and outputs
RUN mkdir -p /app/data /app/checkpoints /app/outputs /app/logs

# Set up non-root user for security
RUN useradd -m -u 1000 nrf && \
    chown -R nrf:nrf /app

USER nrf

# Default command
CMD ["python", "scripts/train.py", "--help"]

# Inference stage (lightweight)
FROM base AS inference

WORKDIR /app

# Install minimal dependencies
RUN pip install \
    torch>=2.0.0 \
    torchvision>=0.15.0 \
    numpy>=1.24.0 \
    pillow>=10.0.0 \
    tqdm>=4.66.0 \
    omegaconf>=2.3.0

# Copy only inference files
COPY src/models/ ./src/models/
COPY src/utils/ ./src/utils/
COPY scripts/sample.py ./scripts/
COPY configs/ ./configs/

# Create directories
RUN mkdir -p /app/checkpoints /app/outputs

# Set up non-root user
RUN useradd -m -u 1000 nrf && \
    chown -R nrf:nrf /app

USER nrf

# Expose port for potential API
EXPOSE 8000

CMD ["python", "scripts/sample.py", "--help"]
