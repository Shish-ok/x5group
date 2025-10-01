
"""
infer_one.py — единичный инференс с постпроцессом
Пример:
python infer_one.py --model runs/rubert_ner --text "вода газированная красная цена 0.5 л"
"""

import argparse, torch, json
from transformers import AutoTokenizer, AutoModelForTokenClassification

try:
    from utils.textnorm import normalize_nfc
    from utils.postprocess import bio_postprocess, merge_units
except Exception:
    from textnorm import normalize_nfc
    from postprocess import bio_postprocess, merge_units

def apply_class_thresholds(logits, id2label, thr):
    probs = torch.softmax(logits, dim=-1)[0]  # [T, C]
    pred_ids = torch.argmax(logits, dim=-1)[0]
    labels=[]
    for i, pid in enumerate(pred_ids.tolist()):
        lab = id2label[pid]
        if lab != "O":
            base = lab.split("-",1)[1]
            if probs[i, pid].item() < thr.get(base, 0.5):
                lab = "O"
        labels.append(lab)
    return labels

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--max-length", type=int, default=128)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(args.model)
    id2label = model.config.id2label

    s = normalize_nfc(args.text)
    enc = tokenizer(s, return_offsets_mapping=True, return_tensors="pt",
                    truncation=True, max_length=args.max_length)
    with torch.no_grad():
        logits = model(**{k:v for k,v in enc.items() if k!="offset_mapping"}).logits

    # срез слабых BRAND и др.
    THR = {"TYPE":0.50, "BRAND":0.78, "VOLUME":0.50, "PERCENT":0.50}
    bio = apply_class_thresholds(logits, id2label, THR)

    # убрать [CLS]/[SEP]
    off = enc["offset_mapping"][0].tolist()
    bio = bio[1:len(off)-1]
    off = off[1:len(off)-1]

    spans = bio_postprocess(bio, off, s)
    spans = merge_units(s, spans)

    print(json.dumps({"text": s, "bio": bio, "spans": spans}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
