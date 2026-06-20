FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt fastapi uvicorn sentence-transformers

COPY . .

EXPOSE 8000

ENV ICM_MODEL_NAME=gpt2
ENV ICM_EMBEDDING_MODEL=all-MiniLM-L6-v2
ENV ICM_MAX_SESSIONS=100
ENV ICM_SESSION_TTL=3600

CMD ["python", "applications/icm_server.py"]
