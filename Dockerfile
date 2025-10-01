FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgomp1 curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements-service.txt ./requirements-service.txt
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install -r requirements-service.txt

RUN python -m pip install --index-url https://download.pytorch.org/whl/cpu torch

COPY ./app ./app

ARG MODEL_RELATIVE_PATH=runs/rubert_base
ENV MODEL_PATH=/models/rubert_base
COPY ./${MODEL_RELATIVE_PATH}/ /models/rubert_base/

RUN test -f /models/rubert_base/model.safetensors -o -f /models/rubert_base/pytorch_model.bin
RUN test -f /models/rubert_base/config.json
RUN ls -1 /models/rubert_base | egrep -q 'tokenizer|vocab|merges|tokenizer\.model|special_tokens_map'

ENV USE_TRANSFORMERS=1 \
    DEVICE=cpu \
    REQUEST_TIMEOUT_S=0.95 \
    WEB_CONCURRENCY=2

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["bash", "-lc", "exec python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers ${WEB_CONCURRENCY} --log-level info --access-log"]
