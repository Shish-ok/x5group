"""
Утилиты подготовки BIO-разметки:
- парсинг и санитайз аннотаций (start, end, tag)
- мердж символных спанов по типу сущности
- токенизация (HF fast tokenizer с offset_mapping или regex-фолбэк)
- проекция спанов на токены → BIO-метки
- сбор одного "record" (text, tokens, offsets, labels)

Зависимости: стандартная библиотека (+ опц. transformers для HF токенайзера).
"""

from __future__ import annotations
import ast
import re
from typing import List, Tuple, Dict, Optional, Iterable, Any
import ast
from collections import Counter

# ----- Константы -----
VALID_KINDS = {"TYPE", "BRAND", "VOLUME", "PERCENT"}
LABEL_LIST = [
    "O",
    "B-TYPE", "I-TYPE",
    "B-BRAND", "I-BRAND",
    "B-VOLUME", "I-VOLUME",
    "B-PERCENT", "I-PERCENT",
]
LABEL2ID: Dict[str, int] = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL: Dict[int, str] = {i: l for l, i in LABEL2ID.items()}

# для regex-токенизации: числа (вкл. дроби), слова (RU/EN, %), отдельные знаки
WORD_RE = re.compile(r"\d+[.,]?\d*|[A-Za-zА-Яа-яЁё%]+|[^\sA-Za-zА-Яа-яЁё0-9]", re.UNICODE)


# ----- Аннотации -----
def parse_ann_cell(cell: Any) -> List[Tuple[int, int, str]]:
    """
    annotation-ячейка: строка со списком кортежей (start, end, tag)
    безопасно парсим через ast.literal_eval
    """
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
    - клип границ в [0, len(text)]
    - '0' → 'O'
    - удаляем 'O'-спаны (для BIO не нужны)
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
    На вход: (start, end, tag) где tag может быть 'B-XXX'/'I-XXX' или 'XXX'.
    Возврат: слитые интервалы (start, end, kind) по kind ∈ VALID_KINDS,
    объединяя пересекающиеся/смежные спаны одного и того же kind.
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


# ----- Токенизация -----
def build_hf_tokenizer(hf_name: str):
    """
    Возвращает HF fast tokenizer (или None, если недоступен/без offset_mapping).
    """
    try:
        from transformers import AutoTokenizer  # type: ignore
        tok = AutoTokenizer.from_pretrained(hf_name, use_fast=True)
        test = tok("тест", return_offsets_mapping=True, add_special_tokens=False)
        if test.get("offset_mapping", None) is None:
            return None
        return tok
    except Exception:
        return None


def tokenize_hf(tok, text: str):
    """
    HF fast tokenizer → (tokens, offsets) без спец-токенов и пустых оффсетов.
    """
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


def tokenize_regex(text: str):
    """
    Regex-токенизация → (tokens, offsets)
    """
    tokens, offsets = [], []
    for m in WORD_RE.finditer(text):
        tokens.append(m.group(0))
        offsets.append((m.start(), m.end()))
    return tokens, offsets


def tokenize_text(text: str, hf_tokenizer=None, mode: str = "auto"):
    """
    Универсальная обёртка:
      - mode='hf' → принудительно HF (ошибка, если недоступен)
      - mode='regex' → принудительно regex
      - mode='auto' → HF если доступен, иначе regex
    """
    if mode not in {"auto", "hf", "regex"}:
        mode = "auto"
    if mode in {"auto", "hf"} and hf_tokenizer is not None:
        try:
            return tokenize_hf(hf_tokenizer, text)
        except Exception:
            if mode == "hf":
                raise
    # fallback
    return tokenize_regex(text)


# ----- BIO-проекция -----
def spans_to_bio_for_tokens(
    offsets: List[Tuple[int, int]],
    spans_merged: List[Tuple[int, int, str]],
) -> List[str]:
    """
    Для каждого спана помечаем первый пересекающийся токен как B-*, остальные как I-*.
    Если токен пересекается с несколькими спанами — сохраняем первую по порядку метку.
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


# ----- Сбор записи -----
def build_record(
    text: str,
    ann_list: List[Tuple[int, int, str]],
    hf_tokenizer=None,
    mode: str = "auto",
    rec_id: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Собирает один record:
      { "id", "text", "tokens", "offsets", "labels" }
    """
    spans_merged = merge_char_spans(ann_list)
    tokens, offsets = tokenize_text(text, hf_tokenizer=hf_tokenizer, mode=mode)
    labels = spans_to_bio_for_tokens(offsets, spans_merged)
    return {
        "id": int(rec_id) if rec_id is not None else None,
        "text": text,
        "tokens": tokens,
        "offsets": offsets,
        "labels": labels,
    }


def build_records(
    texts: Iterable[str],
    anns_iter: Iterable[List[Tuple[int, int, str]]],
    hf_tokenizer=None,
    mode: str = "auto",
    start_id: int = 0,
):
    """
    Пакетная версия build_record. Возвращает список record'ов.
    """
    out = []
    for idx, (t, a) in enumerate(zip(texts, anns_iter), start=start_id):
        out.append(build_record(t, a, hf_tokenizer=hf_tokenizer, mode=mode, rec_id=idx))
    return out


__all__ = [
    "VALID_KINDS", "LABEL_LIST", "LABEL2ID", "ID2LABEL",
    "parse_ann_cell", "sanitize_ann", "merge_char_spans",
    "build_hf_tokenizer", "tokenize_text",
    "spans_to_bio_for_tokens", "build_record", "build_records",
]

