from __future__ import annotations

import os
import json
import asyncio
from datetime import datetime
from typing import Any, Dict, Optional

import aiofiles

class JsonlLogger:
    """
    Простой async-логгер: пишет по 1 JSON-строке на запрос.
    Безопасен внутри процесса (asyncio.Lock).
    Для многопроцессного запуска (несколько воркеров) рекомендуем писать
    в отдельные файлы по PID, чтобы не было "перемешивания строк".
    """

    def __init__(self, path: str):
        self.path = path
        self._lock = asyncio.Lock()

    async def log(self, record: Dict[str, Any]) -> None:
        # создаём каталог при необходимости
        d = os.path.dirname(self.path)
        if d:
            os.makedirs(d, exist_ok=True)

        # сериализация заранее (минимизируем время под локом)
        line = json.dumps(record, ensure_ascii=False)

        async with self._lock:
            async with aiofiles.open(self.path, "a", encoding="utf-8") as f:
                await f.write(line + "\n")


def make_request_record(
    *,
    path: str,
    client_ip: Optional[str],
    input_text: str,
    output_spans: Any,
    latency_ms: float,
    mode: str,
) -> Dict[str, Any]:
    return {
        "ts": datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
        "path": path,
        "client_ip": client_ip,
        "latency_ms": round(latency_ms, 1),
        "mode": mode,                # transformers | stub
        "input": input_text,
        "output": output_spans,
    }