"""
predict_one.py — загрузка модели и предсказание по ОДНОЙ строке.
Выводит JSON со строкой, токенами, BIO-метками и списком сущностей со start/end.

Пример:
python utils/predict_one.py --model runs/rubert_base --text "вода газированная красная цена 0.5 л"
"""

import argparse, json, torch
from typing import List, Dict, Any

# наши утилиты
try:
    from utils.textnorm import normalize_nfc
    from utils.postprocess import bio_postprocess, merge_units
except Exception:
    # если запускаешь из папки utils, импортируй как соседние модули
    from textnorm import normalize_nfc
    from postprocess import bio_postprocess, merge_units

from transformers import AutoTokenizer, AutoModelForTokenClassification

# Пороги уверенности для фильтрации слабых токенов класса
DEFAULT_THR = {"TYPE": 0.55, "BRAND": 0.80, "VOLUME": 0.50, "PERCENT": 0.50}

def apply_thresholds(logits: torch.Tensor, id2label: Dict[int, str], thr: Dict[str, float]) -> List[str]:
    """[1,T,C] -> список меток длины T (с учётом порогов)"""
    probs = torch.softmax(logits, dim=-1)[0]   # [T,C]
    pred  = torch.argmax(logits, dim=-1)[0]    # [T]
    out: List[str] = []
    for t, pid in enumerate(pred.tolist()):
        lab = id2label[pid]
        if lab != "O":
            base = lab.split("-", 1)[1]
            if probs[t, pid].item() < thr.get(base, 0.5):
                lab = "O"
        out.append(lab)
    return out

def predict_one(model_dir: str, text: str, max_length: int, thr: Dict[str, float]) -> Dict[str, Any]:
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    model.eval().to("cuda" if torch.cuda.is_available() else "cpu")

    s = normalize_nfc(text or "")
    enc = tok(
        s,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    with torch.no_grad():
        logits = model(
            input_ids=enc["input_ids"].to(model.device),
            attention_mask=enc["attention_mask"].to(model.device),
            token_type_ids=enc.get("token_type_ids", None).to(model.device) if enc.get("token_type_ids", None) is not None else None,
        ).logits

    id2label = model.config.id2label

    # BIO по всем токенам, затем убираем спец-токены
    bio_all = apply_thresholds(logits, id2label, thr)
    offsets_all = enc["offset_mapping"][0].tolist()
    spec_mask   = enc["special_tokens_mask"][0].tolist()
    input_ids   = enc["input_ids"][0].tolist()

    tokens_plain, offsets_plain, bio_plain = [], [], []
    for iid, off, m, lab in zip(input_ids, offsets_all, spec_mask, bio_all):
        if m == 1:  # спец-токены [CLS]/[SEP]/[PAD]
            continue
        tokens_plain.append(tok.convert_ids_to_tokens(iid))
        offsets_plain.append([int(off[0]), int(off[1])])
        bio_plain.append(lab)

    # Постпроцесс: склейка B-*/I-* в сущности + юниты/проценты по regex
    spans = bio_postprocess(bio_plain, offsets_plain, s)
    spans = merge_units(s, spans)

    # Удобный список сущностей с текстом
    ents = [
        {"start_index": sp["start_index"],
         "end_index": sp["end_index"],
         "entity": sp["entity"],
         "text": s[sp["start_index"]:sp["end_index"]]}
        for sp in spans
    ]

    return {
        "text": s,
        "tokens": tokens_plain,
        "offsets": offsets_plain,
        "bio": bio_plain,
        "spans": spans,      # [{start_index,end_index,entity}]
        "entities": ents     # то же + кусок текста
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="путь к чекпойнту (runs/…)")
    ap.add_argument("--text", required=True, help="входная строка")
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--brand-thr", type=float, default=DEFAULT_THR["BRAND"])
    ap.add_argument("--type-thr", type=float, default=DEFAULT_THR["TYPE"])
    ap.add_argument("--vol-thr", type=float, default=DEFAULT_THR["VOLUME"])
    ap.add_argument("--pct-thr", type=float, default=DEFAULT_THR["PERCENT"])
    args = ap.parse_args()

    thr = {"BRAND": args.brand_thr, "TYPE": args.type_thr, "VOLUME": args.vol_thr, "PERCENT": args.pct_thr}
    res = predict_one(args.model, args.text, args.max_length, thr)
    print(json.dumps(res, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
