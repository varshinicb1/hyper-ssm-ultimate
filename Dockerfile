FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt fastapi uvicorn[standard] sentence-transformers

COPY . .

EXPOSE 8000

ENV ICM_MODEL_NAME=gpt2
ENV ICM_EMBEDDING_MODEL=all-MiniLM-L6-v2
ENV ICM_MAX_SESSIONS=100
ENV ICM_SESSION_TTL=3600
ENV ICM_HOST=0.0.0.0
ENV ICM_PORT=8000
ENV ICM_LOG_LEVEL=INFO

HEALTHCHECK --interval=15s --timeout=5s --retries=3 --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["python", "applications/icm_server.py"]
