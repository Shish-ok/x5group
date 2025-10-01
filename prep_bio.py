"""
prep_bio.py — подготовка BIO-разметки из train.csv

- Читает CSV (по умолчанию: sample, annotation)
- Нормализует текст в Unicode NFC (убирает артефакты типа 'яи' вместо 'й')
- Санитизирует спаны: клип границ, '0'→'O', удаляет 'O'-спаны
- Токенизация: HF fast (ai-forever/ruBert-base) или fallback regex
- BIO-теги для TYPE/BRAND/VOLUME/PERCENT с корректным offset_mapping
- Сохранение JSONL/CoNLL + meta/labels/summary
- (Опц.) train/val split с гарантией наличия всех типов во val

Пример:
  python prep_bio.py --in data/train.csv --out ./x5_bio \
    --tokenizer auto --hf-name ai-forever/ruBert-base --val-size 0.1 --seed 42
"""

from __future__ import annotations
import argparse
import ast
import io
import json
import os
import random
import re
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple, Set, cast

import pandas as pd

# --- нормализация текста ---
try:
    from utils.textnorm import normalize_nfc
except Exception:
    def normalize_nfc(s: str) -> str:
        import unicodedata
        return unicodedata.normalize("NFC", str(s))

# ====== Константы ======
VALID_KINDS: Set[str] = {"TYPE", "BRAND", "VOLUME", "PERCENT"}
LABEL_LIST: List[str] = [
    "O",
    "B-TYPE", "I-TYPE",
    "B-BRAND", "I-BRAND",
    "B-VOLUME", "I-VOLUME",
    "B-PERCENT", "I-PERCENT",
]
LABEL2ID: Dict[str, int] = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL: Dict[int, str] = {i: l for l, i in LABEL2ID.items()}

# Regex-токенизация: числа с дробями, слова RU/EN/% и отдельные символы
WORD_RE = re.compile(r"\d+[.,]?\d*|[A-Za-zА-Яа-яЁё%]+|[^\sA-Za-zА-Яа-яЁё0-9]", re.UNICODE)


def log(msg: str) -> None:
    print(f"[prep_bio] {msg}", flush=True)


# ---------- Аннотации ----------
def parse_ann_cell(cell: Any) -> List[Tuple[int, int, str]]:
    """annotation: строка со списком кортежей (start, end, tag)"""
    try:
        ann = ast.literal_eval(cell) if isinstance(cell, str) else (cell or [])
        if not isinstance(ann, (list, tuple)):
            return []
        out: List[Tuple[int, int, str]] = []
        for item in ann:
            if not isinstance(item, (list, tuple)) or len(item) != 3:
                continue
            s, e, t = item
            out.append((int(s), int(e), str(t)))
        return out
    except Exception:
        return []


def sanitize_ann(text: str, anns: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
    """
    Клип границ в [0, len(text)], '0'→'O', удаляем 'O'-спаны.
    """
    L = len(text)
    cleaned: List[Tuple[int, int, str]] = []
    for s, e, t in anns:
        t = "O" if t == "0" else t
        s = max(0, min(int(s), L))
        e = max(0, min(int(e), L))
        if s >= e:
            continue
        if t == "O":
            continue
        cleaned.append((s, e, t))
    return cleaned


def merge_char_spans(ann_list: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
    """
    На вход: (start, end, tag) где tag 'B-XXX'/'I-XXX' или 'XXX'.
    Выход: слитые интервалы (start, end, kind) по kind ∈ VALID_KINDS,
           объединяя пересекающиеся/смежные спаны одного kind.
    """
    items: List[Tuple[int, int, str]] = []
    for s, e, t in ann_list:
        kind = t.split("-", 1)[1] if "-" in t else t
        if kind in VALID_KINDS:
            items.append((s, e, kind))
    if not items:
        return []
    items.sort(key=lambda x: (x[0], x[1]))
    merged: List[Tuple[int, int, str]] = []
    cs, ce, ck = items[0]
    for s, e, k in items[1:]:
        if k == ck and s <= ce:
            ce = max(ce, e)
        else:
            merged.append((cs, ce, ck))
            cs, ce, ck = s, e, k
    merged.append((cs, ce, ck))
    return merged


# ---------- Токенизация ----------
def build_hf_tokenizer(hf_name: str):
    """Возвращает HF fast tokenizer (или None)."""
    try:
        from transformers import AutoTokenizer  # type: ignore
        tok = AutoTokenizer.from_pretrained(hf_name, use_fast=True)
        test = tok("тест", return_offsets_mapping=True, add_special_tokens=False)
        if test.get("offset_mapping", None) is None:
            return None
        return tok
    except Exception:
        return None


def tokenize_hf(tok, text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    tokens = tok.convert_ids_to_tokens(enc["input_ids"])
    offsets = [(int(s), int(e)) for (s, e) in enc["offset_mapping"]]
    toks2, offs2 = [], []
    for t, (s, e) in zip(tokens, offsets):
        if s == e:
            continue
        toks2.append(t)
        offs2.append((s, e))
    return toks2, offs2


def tokenize_regex(text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    tokens, offsets = [], []
    for m in WORD_RE.finditer(text):
        tokens.append(m.group(0))
        offsets.append((m.start(), m.end()))
    return tokens, offsets


def tokenize_text(text: str, mode: str, hf_tok=None) -> Tuple[List[str], List[Tuple[int, int]]]:
    """
    mode: 'auto'|'hf'|'regex'
    - auto: HF если доступен, иначе regex
    - hf:   только HF (ошибка, если недоступен)
    - regex: только regex
    """
    if mode not in {"auto", "hf", "regex"}:
        mode = "auto"
    if mode in {"auto", "hf"} and hf_tok is not None:
        try:
            return tokenize_hf(hf_tok, text)
        except Exception as e:
            if mode == "hf":
                raise RuntimeError(f"HF токенайзер недоступен: {e}")
    return tokenize_regex(text)


# ---------- BIO-проекция ----------
def spans_to_bio_for_tokens(
    offsets: List[Tuple[int, int]],
    spans_merged: List[Tuple[int, int, str]],
) -> List[str]:
    """
    Для каждого спана помечаем первый пересекающийся токен как B-*, остальные — I-*.
    При пересечении нескольких спанов оставляем первую метку.
    """
    labels = ["O"] * len(offsets)
    for s, e, kind in spans_merged:
        first = True
        for i, (ts, te) in enumerate(offsets):
            if te <= s or ts >= e:
                continue
            new_lab = ("B-" if first else "I-") + kind
            if labels[i] == "O":
                labels[i] = new_lab
            first = False
    return labels


# ---------- Построение записей ----------
def build_record(
    text: str,
    ann_list: List[Tuple[int, int, str]],
    mode: str,
    hf_tok=None,
    rec_id: Optional[int] = None,
) -> Dict[str, Any]:
    # нормализация текста перед любыми операциями
    text_norm = normalize_nfc(text)
    spans_merged = merge_char_spans(ann_list)
    tokens, offsets = tokenize_text(text_norm, mode=mode, hf_tok=hf_tok)
    labels = spans_to_bio_for_tokens(offsets, spans_merged)
    return {
        "id": int(rec_id) if rec_id is not None else None,
        "text": text_norm,
        "tokens": tokens,
        "offsets": offsets,
        "labels": labels,
    }


# ---------- Split ----------
def entity_presence_by_sample(ann_list: List[Tuple[int, int, str]]) -> Set[str]:
    kinds: Set[str] = set()
    for _, _, t in ann_list:
        k = t.split("-", 1)[1] if "-" in t else t
        if k in VALID_KINDS:
            kinds.add(k)
    return kinds


def try_make_split(
    texts: List[str],
    anns: List[List[Tuple[int, int, str]]],
    val_size: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    """
    Делает train/val split по индексам, стараясь обеспечить все VALID_KINDS в val.
    """
    n = len(texts)
    idxs = list(range(n))
    random.Random(seed).shuffle(idxs)
    k = max(1, int(round(n * val_size)))
    val_idx = idxs[:k]
    train_idx = idxs[k:]

    def kinds_in(indices: List[int]) -> Set[str]:
        acc: Set[str] = set()
        for i in indices:
            acc |= entity_presence_by_sample(anns[i])
        return acc

    have = kinds_in(val_idx)
    need = VALID_KINDS - have
    if not need:
        return train_idx, val_idx

    pool = [i for i in train_idx]
    random.Random(seed).shuffle(pool)
    for miss in list(need):
        for i in pool:
            if miss in entity_presence_by_sample(anns[i]):
                val_idx.append(i)
                train_idx.remove(i)
                break
    return train_idx, val_idx


# ---------- Сохранение ----------
def save_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with io.open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def save_conll(path: str, records: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with io.open(path, "w", encoding="utf-8") as f:
        for r in records:
            for tok, lab in zip(r["tokens"], r["labels"]):
                f.write(f"{tok}\t{lab}\n")
            f.write("\n")


def save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with io.open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ---------- Основной процесс ----------
def process(
    in_csv: str,
    out_dir: str,
    text_col: str = "sample",
    ann_col: str = "annotation",
    sep_hint: str = ";",
    tokenizer: str = "auto",                 # 'auto' | 'hf' | 'regex'
    hf_name: str = "ai-forever/ruBert-base",
    val_size: Optional[float] = 0.0,
    seed: int = 42,
    no_split: bool = False,
) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)

    # 1) Чтение CSV
    tried, df = [], None
    for s in [sep_hint, ";", ",", "\t"]:
        tried.append(s)
        try:
            tmp = pd.read_csv(in_csv, sep=s)
            if {text_col, ann_col}.issubset(tmp.columns):
                df = tmp
                sep_hint = s
                break
        except Exception:
            continue
    if df is None:
        raise RuntimeError(f"Не удалось прочитать {in_csv}. Пробовал разделители: {tried}. "
                           f"Ожидаю колонки: {text_col}, {ann_col}")

    texts_raw: List[str] = df[text_col].astype(str).tolist()
    texts: List[str] = [normalize_nfc(t) for t in texts_raw]
    ann_raw: List[List[Tuple[int, int, str]]] = [parse_ann_cell(x) for x in df[ann_col].tolist()]
    ann_list: List[List[Tuple[int, int, str]]] = [sanitize_ann(t, a) for t, a in zip(texts, ann_raw)]

    # 2) Токенайзер
    hf_tok = None
    if tokenizer in {"auto", "hf"}:
        hf_tok = build_hf_tokenizer(hf_name)
        if tokenizer == "hf" and hf_tok is None:
            raise RuntimeError("Запрошен tokenizer=hf, но HF fast токенайзер недоступен.")
    tok_label = f"hf:{hf_name}" if hf_tok is not None else "regex"
    log(f"Токенайзер: {tok_label}")

    # 3) Построение записей
    records: List[Dict[str, Any]] = []
    bad_rows = 0
    mode_effective = ("hf" if hf_tok else "regex") if tokenizer != "auto" else ("hf" if hf_tok else "regex")
    for i, (text, anns) in enumerate(zip(texts, ann_list)):
        rec = build_record(text, anns, mode=mode_effective, hf_tok=hf_tok, rec_id=i)
        if len(rec["tokens"]) != len(rec["labels"]):
            bad_rows += 1
        records.append(rec)

    # 4) Split (опционально)
    do_split = (not no_split) and (val_size is not None) and (val_size > 0.0)
    train_recs, val_recs = records, []
    if do_split:
        vs: float = cast(float, val_size)  # типобезопасно
        train_idx, val_idx = try_make_split(texts, ann_list, vs, seed)
        train_recs = [records[i] for i in train_idx]
        val_recs = [records[i] for i in val_idx]
        log(f"Split: train={len(train_recs)}, val={len(val_recs)} (val_size≈{vs})")

    # 5) Сохранение
    outputs: Dict[str, str] = {}
    out_jsonl = os.path.join(out_dir, "train_bio.jsonl")
    out_conll = os.path.join(out_dir, "train_bio.conll")
    save_jsonl(out_jsonl, records)
    save_conll(out_conll, records)
    outputs["jsonl"] = out_jsonl
    outputs["conll"] = out_conll

    if do_split:
        out_tr_jsonl = os.path.join(out_dir, "train_bio.train.jsonl")
        out_val_jsonl = os.path.join(out_dir, "train_bio.val.jsonl")
        save_jsonl(out_tr_jsonl, train_recs)
        save_jsonl(out_val_jsonl, val_recs)
        outputs["train_jsonl"] = out_tr_jsonl
        outputs["val_jsonl"] = out_val_jsonl

    meta = {
        "tokenizer": tok_label,
        "hf_name": hf_name if hf_tok is not None else None,
        "sep_used": sep_hint,
        "text_col": text_col,
        "ann_col": ann_col,
        "val_size": float(val_size or 0.0),
        "seed": seed,
        "unicode_norm": "NFC",
    }
    save_json(os.path.join(out_dir, "meta.json"), meta)
    save_json(os.path.join(out_dir, "label_list.json"), LABEL_LIST)

    summary = {
        "total_rows": len(records),
        "bad_rows_label_mismatch": bad_rows,
        "tokenizer": tok_label,
        "outputs": outputs,
    }
    save_json(os.path.join(out_dir, "bio_summary.json"), summary)
    log(json.dumps(summary, ensure_ascii=False))
    return summary


# ---------- CLI ----------
def main():
    p = argparse.ArgumentParser(description="Подготовка BIO-разметки из train.csv")
    p.add_argument("--in", dest="in_csv", required=True, help="Путь к train.csv")
    p.add_argument("--out", dest="out_dir", required=True, help="Каталог для сохранения")
    p.add_argument("--text-col", default="sample", help="Колонка с текстом (sample)")
    p.add_argument("--ann-col", default="annotation", help="Колонка с аннотациями (annotation)")
    p.add_argument("--sep", dest="sep_hint", default=";", help="Разделитель CSV (по умолчанию ';')")
    p.add_argument("--tokenizer", choices=["auto", "hf", "regex"], default="auto",
                   help="auto (HF если доступен, иначе regex) | hf | regex")
    p.add_argument("--hf-name", default="ai-forever/ruBert-base", help="Имя/путь HF токенайзера")
    p.add_argument("--val-size", type=float, default=0.0, help="Доля валидации (0..1), 0 — без split")
    p.add_argument("--seed", type=int, default=42, help="Seed сплита")
    p.add_argument("--no-split", action="store_true", help="Не делать train/val split даже если val-size>0")

    args = p.parse_args()

    try:
        process(
            in_csv=args.in_csv,
            out_dir=args.out_dir,
            text_col=args.text_col,
            ann_col=args.ann_col,
            sep_hint=args.sep_hint,
            tokenizer=args.tokenizer,
            hf_name=args.hf_name,
            val_size=args.val_size,
            seed=args.seed,
            no_split=args.no_split,
        )
    except Exception as e:
        log(f"Ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
