# HeartMuLa RunPod Serverless - Self-contained with weights
# Just build and deploy - no external volume needed

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/ckpt
ENV HF_HOME=/app/huggingface
ENV TRANSFORMERS_CACHE=/app/huggingface

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (for better caching)
COPY runpod/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Install huggingface CLI for downloading models
RUN pip install --no-cache-dir huggingface_hub[cli]

# Download model weights during build (this makes the image large but self-contained)
RUN mkdir -p /app/ckpt && \
    huggingface-cli download HeartMuLa/HeartMuLaGen --local-dir /app/ckpt && \
    huggingface-cli download HeartMuLa/HeartMuLa-oss-3B --local-dir /app/ckpt/HeartMuLa-oss-3B && \
    huggingface-cli download HeartMuLa/HeartCodec-oss --local-dir /app/ckpt/HeartCodec-oss

# Copy the heartlib source code
COPY src/ /app/src/
COPY pyproject.toml /app/pyproject.toml

# Install heartlib
RUN pip install -e /app/

# Copy the handler
COPY runpod/handler.py /app/handler.py

# Verify weights exist
RUN ls -la /app/ckpt/ && \
    ls -la /app/ckpt/HeartMuLa-oss-3B/ && \
    ls -la /app/ckpt/HeartCodec-oss/

# Entry point
CMD ["python", "-u", "/app/handler.py"]
