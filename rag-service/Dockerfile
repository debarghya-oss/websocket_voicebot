FROM python:3.11-slim-bookworm

# System deps — Linux native
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libmagic1 \
    poppler-utils \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /rag-service

# Dependencies first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download embedding model into the image so first startup is instant
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('BAAI/bge-small-en-v1.5')"

COPY app/ ./app/

RUN mkdir -p /uploads

EXPOSE 8000

# 2 workers — embedder is lru_cached per process, safe to parallelise
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]