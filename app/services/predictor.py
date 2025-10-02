from __future__ import annotations

import asyncio
import logging
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

from app.core.config import settings

logger = logging.getLogger("ner")

Span = Dict[str, int | str]
_WORD_RE = re.compile(r"\w+|[^\w\s]+", flags=re.UNICODE)


class _StubNer:
    def predict(self, text: str) -> List[Span]:
        if not text:
            return []
        spans: List[Span] = []
        words = list(_WORD_RE.finditer(text))
        in_seq = False
        for m in words:
            tag = "I-TYPE" if in_seq else "B-TYPE"
            spans.append({"start_index": m.start(), "end_index": m.end(), "entity": tag})
            in_seq = True
        return spans


def _label_to_ent(label: str) -> str | None:
    if label == "O":
        return None
    if "-" in label:
        _, ent = label.split("-", 1)
        return ent
    return label  # fallback


class _TfNer:
    def __init__(self, model_path: Optional[str], model_name: Optional[str], device_hint: Optional[str]):
        from transformers import AutoTokenizer, AutoModelForTokenClassification  # type: ignore
        import torch  # type: ignore

        src = model_path or model_name
        if not src:
            raise ValueError("Either model_path or model_name must be provided")

        self.tokenizer = AutoTokenizer.from_pretrained(src)  # type: ignore[arg-type]
        self.model = AutoModelForTokenClassification.from_pretrained(src)  # type: ignore[arg-type]
        self.model.eval()

        device_str = device_hint or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device_str)
        self.model.to(self.device)

        self.id2label = self.model.config.id2label
        logger.info("NER mode: transformers | src=%s | device=%s | labels=%s",
                    src, self.device, list(self.id2label.values()))

    def predict(self, text: str) -> List[Span]:
        if not text.strip():
            return []

        import torch

        with torch.inference_mode():
            enc = self.tokenizer(
                text,
                return_offsets_mapping=True,
                return_tensors="pt",
                truncation=True,
            )
            offsets = enc.pop("offset_mapping")[0].tolist()
            inputs = {k: v.to(self.device) for k, v in enc.items()}

            logits = self.model(**inputs).logits[0]
            pred_ids = logits.argmax(-1).tolist()

        sub = []
        for (s, e), pid in zip(offsets, pred_ids):
            if e == 0:
                continue
            sub.append(((int(s), int(e)), self.id2label[pid]))
        spans: List[Span] = []
        words = list(_WORD_RE.finditer(text))
        prev_ent: Optional[str] = None

        for w in words:
            ws, we = w.start(), w.end()

            labs: List[str] = []
            for (ts, te), lab in sub:
                if te <= ws or ts >= we:
                    continue
                labs.append(lab)

            ents = [_label_to_ent(l) for l in labs if l != "O"]
            ents = [e for e in ents if e is not None]
            if not ents:
                prev_ent = None
                continue

            ent = Counter(ents).most_common(1)[0][0]
            has_b = any(l.startswith("B-") and _label_to_ent(l) == ent for l in labs)

            if prev_ent is None or ent != prev_ent or has_b:
                tag = f"B-{ent}"
            else:
                tag = f"I-{ent}"

            spans.append({"start_index": ws, "end_index": we, "entity": tag})
            prev_ent = ent

        return spans


class NerModel:
    def __init__(self, _unused_model_path: Optional[str] = None):
        self.impl = None
        if settings.use_transformers and (settings.model_path or settings.model_name):
            try:
                self.impl = _TfNer(settings.model_path, settings.model_name, settings.device)
            except Exception as e:
                logger.exception("Failed to init transformers backend")
                raise RuntimeError(f"Transformers init failed: {e}") from e
        else:
            logger.warning("NER mode: stub (use_transformers=%s, model_path=%s, model_name=%s)",
                           settings.use_transformers, settings.model_path, settings.model_name)
            self.impl = _StubNer()

    def predict(self, text: str) -> List[Span]:
        return self.impl.predict(text)

    def info(self) -> dict:
        impl = getattr(self, "impl", None)
        if impl is None:
            return {"mode": "unknown"}

        if isinstance(impl, _TfNer):
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