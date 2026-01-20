# HeartMuLa RunPod Serverless - Lightweight build
# Models downloaded at RUNTIME, not build time

FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/ckpt
ENV DEBIAN_FRONTEND=noninteractive

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA
RUN pip install --no-cache-dir \
    torch==2.4.1 \
    torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu124

# Copy and install requirements (without torch since already installed)
COPY runpod/requirements-minimal.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy source code
COPY src/ /app/src/
COPY pyproject.toml /app/pyproject.toml

# Install heartlib
RUN pip install --no-cache-dir -e /app/

# Copy handler
COPY runpod/handler.py /app/handler.py

# Create dirs
RUN mkdir -p /app/ckpt

CMD ["python", "-u", "/app/handler.py"]
