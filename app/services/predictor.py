from __future__ import annotations

import asyncio
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

from app.core.config import settings

Span = Dict[str, int | str]
_token_re = re.compile(r"\S+")

class _StubNer:
    def predict(self, text: str) -> List[Span]:
        spans: List[Span] = []
        if not text:
            return spans
        for i, m in enumerate(_token_re.finditer(text)):
            start = m.start()
            end = m.end()
            tag = "B-TYPE" if i == 0 else "I-TYPE"
            spans.append({"start_index": start, "end_index": end, "entity": tag})
        return spans


class _TfNer:
    def __init__(self, model_path: Optional[str], model_name: Optional[str], device_hint: Optional[str]):
        from transformers import AutoTokenizer, AutoModelForTokenClassification  # type: ignore
        import torch  # type: ignore

        self.tokenizer = AutoTokenizer.from_pretrained(model_path or model_name)  # type: ignore[arg-type]
        self.model = AutoModelForTokenClassification.from_pretrained(model_path or model_name)  # type: ignore[arg-type]
        self.model.eval()

        if device_hint:
            self.device = torch.device(device_hint)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.id2label = self.model.config.id2label

    def predict(self, text: str) -> List[Span]:
        if not text:
            return []

        import torch  # type: ignore

        with torch.inference_mode():
            enc = self.tokenizer(
                text,
                return_offsets_mapping=True,
                return_tensors="pt",
                truncation=True,
            )
            offsets = enc["offset_mapping"][0].tolist()
            inputs = {k: v.to(self.device) for k, v in enc.items() if k != "offset_mapping"}
            logits = self.model(**inputs).logits[0]
            pred_ids = logits.argmax(-1).tolist()

        # собираем BIO-спаны по символам (inclusive end)
        spans: List[Span] = []
        current: Optional[List[int | str]] = None

        for (start, end), pid in zip(offsets, pred_ids):
            if end == 0:
                continue
            label = self.id2label[pid]

            if label == "O":
                if current:
                    spans.append({"start_index": int(current[0]), "end_index": int(current[1]) - 1, "entity": str(current[2])})
                    current = None
                continue

            if label.startswith("B-"):
                if current:
                    spans.append({"start_index": int(current[0]), "end_index": int(current[1]) - 1, "entity": str(current[2])})
                ent = label.split("-", 1)[1]
                current = [start, end, f"B-{ent}"]

            elif label.startswith("I-"):
                ent = label.split("-", 1)[1]
                if current is None:
                    current = [start, end, f"B-{ent}"]
                else:
                    current[1] = end  # продолжаем текущий спан

            else:
                if current:
                    spans.append({"start_index": int(current[0]), "end_index": int(current[1]) - 1, "entity": str(current[2])})
                current = [start, end, "B-TYPE"]

        if current:
            spans.append({"start_index": int(current[0]), "end_index": int(current[1]) - 1, "entity": str(current[2])})

        return spans


class NerModel:
    def __init__(self, _unused_model_path: Optional[str] = None):
        self.impl = None
        if settings.use_transformers and (settings.model_path or settings.model_name):
            try:
                self.impl = _TfNer(settings.model_path, settings.model_name, settings.device)
            except Exception as e:
                print(f"[warn] transformers init failed: {e}. Falling back to stub.")
                self.impl = _StubNer()
        else:
            self.impl = _StubNer()

    def predict(self, text: str) -> List[Span]:
        return self.impl.predict(text)

    def info(self) -> dict:
        impl = getattr(self, "impl", None)
        if impl is None:
            return {"mode": "unknown"}

        if impl.__class__.__name__ == "_TfNer":
            return {
                "mode": "transformers",
                "model_path": getattr(settings, "model_path", None),
                "model_name": getattr(settings, "model_name", None),
                "device": str(getattr(impl, "device", "cpu")),
                "labels": list(getattr(impl, "id2label", {}).values()),
            }

        return {"mode": "stub"}


class AsyncPredictor:
    def __init__(self, model: NerModel, max_workers: int = 2):
        self.model = model
        self._pool = ThreadPoolExecutor(max_workers=max_workers)

    async def predict(self, text: str) -> List[Span]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._pool, self.model.predict, text)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    def info(self) -> dict:
        return self.model.info()