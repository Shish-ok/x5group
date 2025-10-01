"""
train_ner.py — базовое обучение RuBERT для BIO с кросс-версийной совместимостью transformers.

Вход: JSONL от prep_bio.py: [{"id","text","tokens","offsets","labels"}]
Выход:
  - чекпойнт модели в --output-dir (best, если поддерживается версия)
  - pred_bio.val.jsonl (BIO на валидации)
  - metrics_history.json / best_metrics.json
"""

from __future__ import annotations
import argparse, io, json, os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional, Set, cast

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    TrainingArguments, Trainer, DataCollatorForTokenClassification,
    set_seed, __version__ as HF_VER
)
from inspect import signature
from seqeval.metrics import f1_score, precision_score, recall_score

# --- NFC нормализация ---
try:
    from utils.textnorm import normalize_nfc
except Exception:
    import unicodedata
    def normalize_nfc(s: str) -> str:
        return unicodedata.normalize("NFC", str(s))

VALID_KINDS: Set[str] = {"TYPE","BRAND","VOLUME","PERCENT"}

# ---------------- I/O ----------------
def read_jsonl(path: str) -> List[dict]:
    rows=[]
    with io.open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln=ln.strip()
            if ln: rows.append(json.loads(ln))
    return rows

# ------------- BIO utils -------------
def bio_to_token_spans(labels: List[str]) -> List[Tuple[int,int,str]]:
    spans=[]; start=None; cur=None
    for i, lab in enumerate(labels):
        if not lab or lab=="O":
            if start is not None and cur is not None: spans.append((start,i,cur))
            start=None; cur=None; continue
        if "-" in lab: pref, kind = lab.split("-",1)
        else: pref, kind = "B", lab
        if pref=="B":
            if start is not None and cur is not None: spans.append((start,i,cur))
            start=i; cur=kind
        elif pref=="I":
            if start is None or cur!=kind:
                if start is not None and cur is not None: spans.append((start,i,cur))
                start=i; cur=kind
        else:
            if start is not None and cur is not None: spans.append((start,i,cur))
            start=None; cur=None
    if start is not None and cur is not None: spans.append((start,len(labels),cur))
    return spans

def token_spans_to_char_spans(spans_tok: List[Tuple[int,int,str]],
                              offsets: List[Tuple[int,int]],
                              text_len: int) -> List[Tuple[int,int,str]]:
    out=[]
    for ti,tj,k in spans_tok:
        head=None; tail=None
        for t in range(ti,tj):
            s,e=offsets[t]
            if s!=e: head=(s,e); break
        for t in range(tj-1,ti-1,-1):
            s,e=offsets[t]
            if s!=e: tail=(s,e); break
        if head is None or tail is None: continue
        s=max(0,min(head[0],text_len)); e=max(0,min(tail[1],text_len))
        if s<e: out.append((s,e,k))
    return out

def spans_to_token_bio(offsets: List[Tuple[int,int]], spans_char: List[Tuple[int,int,str]]) -> List[str]:
    spans_char=sorted(spans_char,key=lambda x:(x[2],x[0],x[1]))
    merged=[]
    if spans_char:
        cs,ce,ck=spans_char[0]
        for s,e,k in spans_char[1:]:
            if k==ck and s<=ce: ce=max(ce,e)
            else: merged.append((cs,ce,ck)); cs,ce,ck=s,e,k
        merged.append((cs,ce,ck))
    labels=["O"]*len(offsets)
    for s,e,k in merged:
        first=True
        for i,(ts,te) in enumerate(offsets):
            if te<=s or ts>=e: continue
            labels[i]=("B-" if first else "I-")+k
            first=False
    return labels

def load_label_list(data_dir: str) -> List[str]:
    p = os.path.join(data_dir, "label_list.json")
    with io.open(p, "r", encoding="utf-8") as f:
        labels = json.load(f)
    return labels

# --------- dataset/encoding ----------
def build_hf_dataset(jsonl_path: str, tokenizer, label2id: Dict[str,int], max_length: int):
    rows = read_jsonl(jsonl_path)
    ds = Dataset.from_list(rows)

    def encode(example):
        text = normalize_nfc(example["text"])
        # Включаем спец-токены сразу, чтобы сразу выровнять метки под input_ids
        enc = tokenizer(
            text,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            truncation=True,
            max_length=max_length,
            padding=False,
            add_special_tokens=True,
        )
        offsets_all = enc["offset_mapping"]              # длина = len(input_ids)
        spec_mask   = enc["special_tokens_mask"]         # 1 на спец-токенах
        # offsets без спецов — на них и будем проецировать BIO
        new_offsets = [(int(s),int(e)) for (s,e),m in zip(offsets_all, spec_mask) if m==0]

        old_offsets = [(int(s),int(e)) for (s,e) in example["offsets"]]
        old_labels  = [str(x) for x in example["labels"]]

        if len(new_offsets)==len(old_offsets) and all(a==b for a,b in zip(new_offsets, old_offsets)):
            token_labels = old_labels
        else:
            tok_spans  = bio_to_token_spans(old_labels)
            char_spans = token_spans_to_char_spans(tok_spans, old_offsets, len(text))
            token_labels = spans_to_token_bio(new_offsets, char_spans)

        # Выравниваем метки по input_ids: на спец-токены ставим -100
        aligned: List[int] = []
        i_plain = 0
        for m in spec_mask:
            if m == 1:
                aligned.append(-100)
            else:
                aligned.append(label2id.get(token_labels[i_plain], label2id["O"]))
                i_plain += 1

        enc["labels"] = aligned
        # больше offsets/special_tokens_mask не нужны
        enc.pop("offset_mapping", None)
        enc.pop("special_tokens_mask", None)
        return enc

    ds = ds.map(encode, remove_columns=[c for c in ds.column_names if c not in {"id"}],
                desc=f"Tokenizing {os.path.basename(jsonl_path)}")
    return ds

@dataclass
class CollatorWithLabels(DataCollatorForTokenClassification):
    pad_to_multiple_of: Optional[int] = None
    def __call__(self, features):
        labels = [f.pop("labels") for f in features]  # уже выровнены под input_ids
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        input_ids = cast(torch.Tensor, batch["input_ids"])
        seq_len = int(input_ids.shape[1])
        # паддинг меток -100 справа
        padded=[]
        for lr in labels:
            if len(lr) < seq_len:
                lr = lr + [-100]*(seq_len - len(lr))
            elif len(lr) > seq_len:
                lr = lr[:seq_len]
            padded.append(lr)
        batch["labels"] = torch.tensor(padded, dtype=torch.long)
        return batch

# --- кросс-версионный билдер TrainingArguments ---
def build_training_args(args) -> TrainingArguments:
    kw = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_train_epochs=args.epochs,
        warmup_ratio=0.1,
        fp16=True,
        gradient_accumulation_steps=args.grad_accum,
        logging_steps=50,
        report_to=["none"],
    )
    try:
        supported = set(signature(TrainingArguments.__init__).parameters.keys())
    except Exception:
        supported = set(kw.keys())

    # выставляем стратегии, если поддерживаются
    if "evaluation_strategy" in supported:
        kw["evaluation_strategy"] = "epoch"
    elif "evaluate_during_training" in supported:
        kw["evaluate_during_training"] = True

    if "save_strategy" in supported:
        kw["save_strategy"] = "epoch"
    elif "save_steps" in supported:
        kw["save_steps"] = 500

    # включаем best-only, если возможно и стратегии совпадают
    if (
        "load_best_model_at_end" in supported
        and kw.get("evaluation_strategy", None) in ("epoch", "steps")
        and kw.get("save_strategy", None) == kw.get("evaluation_strategy", None)
    ):
        kw["load_best_model_at_end"] = True
        if "metric_for_best_model" in supported:
            kw["metric_for_best_model"] = "f1"
        if "greater_is_better" in supported:
            kw["greater_is_better"] = True

    # чистим неподдерживаемое
    for k in list(kw.keys()):
        if k not in supported:
            kw.pop(k, None)

    return TrainingArguments(**kw)

# --------------- train ---------------
def save_pred_jsonl(path: str, ids: List[Any], preds: List[List[int]], id2label: Dict[int,str]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out=[]
    for rid, label_ids in zip(ids, preds):
        labs=[id2label[int(l)] for l in label_ids]
        out.append({"id": rid, "labels": labs})
    with io.open(path, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False)+"\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model-name", default="ai-forever/ruBert-base")
    ap.add_argument("--train-file", default="train_bio.train.jsonl")
    ap.add_argument("--val-file",   default="train_bio.val.jsonl")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grad-accum", type=int, default=1)
    args = ap.parse_args()

    print(f"[train_ner] transformers={HF_VER}")
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    label_list = load_label_list(args.data_dir)
    label2id = {l:i for i,l in enumerate(label_list)}
    id2label = {i:l for l,i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )

    train_path = os.path.join(args.data_dir, args.train_file)
    val_path   = os.path.join(args.data_dir, args.val_file)
    if not os.path.isfile(val_path):
        train_path = os.path.join(args.data_dir, "train_bio.jsonl")
        val_path   = os.path.join(args.data_dir, "train_bio.jsonl")

    ds_train = build_hf_dataset(train_path, tokenizer, label2id, args.max_length)
    ds_val   = build_hf_dataset(val_path, tokenizer, label2id, args.max_length)
    ids_val  = [r["id"] for r in read_jsonl(val_path)]

    collator = CollatorWithLabels(tokenizer=tokenizer)

    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=-1)
        labels = p.label_ids
        true_labels = []
        true_preds  = []
        for l_row, p_row in zip(labels, preds):
            tl=[]; tp=[]
            for y, yhat in zip(l_row, p_row):
                if y == -100: 
                    continue
                tl.append(label_list[y])
                tp.append(label_list[yhat])
            true_labels.append(tl)
            true_preds.append(tp)
        return {
            "precision": precision_score(true_labels, true_preds),
            "recall": recall_score(true_labels, true_preds),
            "f1": f1_score(true_labels, true_preds)
        }

    args_tr = build_training_args(args)
    print("[train_ner] eval_strategy:", getattr(args_tr, "evaluation_strategy", None),
          "save_strategy:", getattr(args_tr, "save_strategy", None),
          "load_best:", getattr(args_tr, "load_best_model_at_end", None))

    trainer = Trainer(
        model=model,
        args=args_tr,
        data_collator=collator,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # --- отчёты метрик ---
    with io.open(os.path.join(args.output_dir, "metrics_history.json"), "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, ensure_ascii=False, indent=2)

    best = {}
    for log in trainer.state.log_history:
        if "eval_f1" in log and (not best or log["eval_f1"] > best.get("eval_f1", -1)):
            best = log
    with io.open(os.path.join(args.output_dir, "best_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)

    # --- BIO-предсказания на валидации ---
    pred_out = trainer.predict(cast(Any, ds_val))
    pred_ids = np.argmax(pred_out.predictions, axis=-1)
    assert pred_out.label_ids is not None
    label_ids = cast(np.ndarray, pred_out.label_ids)

    pred_per_sample: List[List[int]] = []
    for row_pred, row_gold in zip(pred_ids, label_ids):
        seq = [int(p) for p, y in zip(row_pred, row_gold) if y != -100]
        pred_per_sample.append(seq)

    save_pred_jsonl(os.path.join(args.output_dir,"pred_bio.val.jsonl"), ids_val, pred_per_sample, id2label)

    print("\n✓ Обучение завершено. Модель сохранена в:", args.output_dir)
    print("✓ Вал-предсказания (BIO):", os.path.join(args.output_dir,"pred_bio.val.jsonl"))
    print("✓ Отчёты:", os.path.join(args.output_dir,"metrics_history.json"),
          "и", os.path.join(args.output_dir,"best_metrics.json"))

if __name__ == "__main__":
    main()

