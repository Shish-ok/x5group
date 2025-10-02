from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import os
from fastapi import APIRouter, HTTPException, Request
from app.schemas.predict import PredictIn, PredictOut
from app.core.config import settings
from app.utils.reqlog import make_request_record

import time
from typing import List, Literal
from pydantic import BaseModel, Field

try:
    import aiofiles  # для неблокирующей записи
except Exception:     # если не установлено — запишем синхронно через executor
    aiofiles = None

_REQUEST_LOG_ENABLED = os.getenv("REQUEST_LOG_ENABLED", "1") == "1"
_REQUEST_LOG_DIR = os.getenv("REQUEST_LOG_DIR", "/var/log/ner")
_REQUEST_LOG_PATH = os.path.join(_REQUEST_LOG_DIR, f"requests-{os.getpid()}.jsonl")

_INFER_SEM = asyncio.Semaphore(int(os.getenv("MAX_INFLIGHT", "2")))
# Сколько ждём свободного слота перед 429 (по умолчанию 10 мс)
_ACQUIRE_TIMEOUT = float(os.getenv("ACQUIRE_TIMEOUT_MS", "10")) / 1000.0

def _write_line_sync(path: str, line: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")

router = APIRouter()

@router.post("/predict", response_model=PredictOut)
async def predict(body: PredictIn, request: Request) -> PredictOut:
    predictor = request.app.state.predictor
    t0 = time.perf_counter()

    # 1) быстро пробуем занять слот; если очередь — сразу 429
    try:
        await asyncio.wait_for(_INFER_SEM.acquire(), timeout=_ACQUIRE_TIMEOUT)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=429, detail="busy: try later")

    try:
        # 2) собственно инференс с жёстким дедлайном
        result = await asyncio.wait_for(
            predictor.predict(body.input),
            timeout=settings.request_timeout_s,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Prediction timeout")
    finally:
        _INFER_SEM.release()

    # 3) лог в JSONL (по строке на запрос), путь смонтирован на хост
    if _REQUEST_LOG_ENABLED:
        rec = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "path": "/api/predict",
            "client_ip": request.client.host if request.client else None,
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 1),
            "mode": predictor.info().get("mode", "unknown"),
            "input": body.input,
            "output": result,
        }
        line = json.dumps(rec, ensure_ascii=False)
        if aiofiles:
            os.makedirs(os.path.dirname(_REQUEST_LOG_PATH), exist_ok=True)
            async with aiofiles.open(_REQUEST_LOG_PATH, "a", encoding="utf-8") as f:
                await f.write(line + "\n")
        else:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, _write_line_sync, _REQUEST_LOG_PATH, line)

    return result