"""
predict_csv.py — батчевый инференс и сбор submission.csv

Пример:
python predict_csv.py --model runs/rubert_base --in-csv data/test.csv --out-csv submission.csv \
  --id-col id --text-col sample --max-length 128 --batch-size 64
"""

import argparse, io, json, os
from typing import List, Dict, Any, Tuple
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification

# наши утилиты
try:
    from utils.textnorm import normalize_nfc
    from utils.postprocess import bio_postprocess, merge_units
except Exception:
    # на случай запуска из другого cwd
    from textnorm import normalize_nfc
    from postprocess import bio_postprocess, merge_units

# пороги уверенности (срежет слабые BRAND)
THR = {"TYPE":0.50, "BRAND":0.78, "VOLUME":0.50, "PERCENT":0.50}

def labels_with_thresholds(logits: torch.Tensor, id2label: Dict[int, str]) -> List[List[str]]:
    """logits: [B, T, C] -> BIO-метки с порогами по классам."""
    probs = torch.softmax(logits, dim=-1)          # [B,T,C]
    pred_ids = torch.argmax(logits, dim=-1)        # [B,T]
    out: List[List[str]] = []
    B, T = pred_ids.shape
    for b in range(B):
        row=[]
        for t in range(T):
            pid = int(pred_ids[b,t].item())
            lab = id2label[pid]
            if lab != "O":
                base = lab.split("-",1)[1]
                if probs[b,t,pid].item() < THR.get(base, 0.5):
                    lab = "O"
            row.append(lab)
        out.append(row)
    return out

def to_submission_str(spans: List[Dict[str, Any]]) -> str:
    """[(s,e,'TYPE'),...] -> строка, как в train.csv (list of tuples)."""
    tuples = [(int(s["start_index"]), int(s["end_index"]), str(s["entity"])) for s in spans]
    return str(tuples)

def infer_batch(model, tokenizer, texts: List[str], max_length: int) -> List[List[Dict[str, Any]]]:
    """Возвращает для каждого текста список символьных спанов."""
    enc = tokenizer(
        texts,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    dev = next(model.parameters()).device
    with torch.no_grad():
        logits = model(
            input_ids=enc["input_ids"].to(dev),
            attention_mask=enc["attention_mask"].to(dev),
            token_type_ids=enc.get("token_type_ids", None).to(dev) if enc.get("token_type_ids", None) is not None else None,
        ).logits

    id2label = model.config.id2label
    bio_all = labels_with_thresholds(logits, id2label)

    spans_all: List[List[Dict[str, Any]]] = []
    for i, bio in enumerate(bio_all):
        offs = enc["offset_mapping"][i].tolist()
        spec = enc["special_tokens_mask"][i].tolist()
        # срезаем спец-токены
        plain_offsets = [tuple(map(int, off)) for off, m in zip(offs, spec) if m == 0]
        plain_bio     = [lab for lab, m in zip(bio, spec) if m == 0]
        text = texts[i]
        spans = bio_postprocess(plain_bio, plain_offsets, text)
        spans = merge_units(text, spans)
        spans_all.append(spans)
    return spans_all

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--in-csv", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--id-col", default="id")
    ap.add_argument("--text-col", default="sample")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=128)
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv, delimiter = ";")
    if args.id_col not in df.columns or args.text_col not in df.columns:
        raise ValueError(f"В {args.in_csv} нет колонок {args.id_col}/{args.text_col}")

    # нормализация и подготовка
    ids = df[args.id_col].tolist()
    texts = [normalize_nfc(str(x)) for x in df[args.text_col].astype(str).tolist()]

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(args.model)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev).eval()

    # инференс батчами
    spans_all: List[List[Dict[str, Any]]] = []
    for i in range(0, len(texts), args.batch_size):
        batch = texts[i:i+args.batch_size]
        spans_all.extend(infer_batch(model, tokenizer, batch, args.max_length))

    # собираем сабмит
    out_rows = []
    for rid, text, spans in zip(ids, texts, spans_all):
        out_rows.append({args.id_col: rid, "annotation": to_submission_str(spans)})

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    pd.DataFrame(out_rows).to_csv(args.out_csv, index=False)
    print(f"[predict_csv] Saved: {args.out_csv}  (rows={len(out_rows)})")

if __name__ == "__main__":
    main()
