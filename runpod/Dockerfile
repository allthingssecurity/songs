# HeartMuLa RunPod Serverless
# Downloads model weights at runtime (first cold start)

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

# Copy and install requirements
COPY runpod/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Install huggingface CLI
RUN pip install --no-cache-dir huggingface_hub[cli]

# Copy the heartlib source code
COPY src/ /app/src/
COPY pyproject.toml /app/pyproject.toml

# Install heartlib
RUN pip install --no-cache-dir -e /app/

# Copy the handler (with runtime model download)
COPY runpod/handler.py /app/handler.py

# Create checkpoint directory
RUN mkdir -p /app/ckpt

# Entry point
CMD ["python", "-u", "/app/handler.py"]
