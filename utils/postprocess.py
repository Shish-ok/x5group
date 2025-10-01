from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional, Set
import re

VALID_KINDS: Set[str] = {"TYPE", "BRAND", "VOLUME", "PERCENT"}

# стоп-слова, после которых TYPE обрезаем (в gold «хвост» часто O)
STOP_INSIDE_TYPE = {
    "для","в","во","с","со","без","на","по","от","из","под","над","при","к","о","об","обо","про"
}

# Регексы для юнитов/процентов (можно использовать после BIO-постпроцесса)
RE_VOL = re.compile(r'(\d+[.,]?\d*)\s?(мл|л|г|кг|шт)\b', re.I | re.U)
RE_PCT = re.compile(r'(\d+[.,]?\d*)\s?%', re.U)

def _bio_to_token_spans(labels: List[str]) -> List[Tuple[int,int,str]]:
    """BIO -> (start_tok, end_tok_exclusive, kind); I без B стартует новую сущность."""
    spans: List[Tuple[int,int,str]] = []
    start: Optional[int] = None
    cur: Optional[str] = None
    for i, lab in enumerate(labels):
        if not lab or lab == "O":
            if start is not None and cur is not None:
                spans.append((start, i, cur))
            start, cur = None, None
            continue
        if "-" in lab:
            pref, k = lab.split("-", 1)
        else:
            pref, k = "B", lab
        if k not in VALID_KINDS:
            if start is not None and cur is not None:
                spans.append((start, i, cur))
            start, cur = None, None
            continue
        if pref == "B":
            if start is not None and cur is not None:
                spans.append((start, i, cur))
            start, cur = i, k
        elif pref == "I":
            if start is None or cur != k:
                if start is not None and cur is not None:
                    spans.append((start, i, cur))
                start, cur = i, k
        else:
            if start is not None and cur is not None:
                spans.append((start, i, cur))
            start, cur = None, None
    if start is not None and cur is not None:
        spans.append((start, len(labels), cur))
    return spans

def _first_valid_offset(offsets: List[Tuple[int,int]], i: int, j: int) -> Optional[Tuple[int,int]]:
    for k in range(i, j):
        s, e = offsets[k]
        if s != e:
            return s, e
    return None

def _last_valid_offset(offsets: List[Tuple[int,int]], i: int, j: int) -> Optional[Tuple[int,int]]:
    for k in range(j-1, i-1, -1):
        s, e = offsets[k]
        if s != e:
            return s, e
    return None

def _trim_spaces(text: str, s: int, e: int) -> Tuple[int,int]:
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e-1].isspace():
        e -= 1
    return s, e

def _shrink_type_by_stopwords(text: str, s: int, e: int) -> Tuple[int,int]:
    """Обрезает TYPE перед служебным словом (если оно входит в спан)."""
    sub = text[s:e]
    pos = 0
    for part in sub.split():
        ps = sub.find(part, pos)
        if ps < 0:
            break
        pe = ps + len(part)
        if part.lower() in STOP_INSIDE_TYPE:
            return s, s + ps
        pos = pe
    return s, e

def _merge_same_type_char_spans(spans: List[Tuple[int,int,str]]) -> List[Tuple[int,int,str]]:
    if not spans:
        return spans
    spans = sorted(spans, key=lambda x: (x[2], x[0], x[1]))
    out: List[Tuple[int,int,str]] = []
    cs, ce, ck = spans[0]
    for s, e, k in spans[1:]:
        if k == ck and s <= ce:
            ce = max(ce, e)
        else:
            out.append((cs, ce, ck))
            cs, ce, ck = s, e, k
    out.append((cs, ce, ck))
    return out

def _drop_overlaps_by_longest(spans: List[Tuple[int,int,str]]) -> List[Tuple[int,int,str]]:
    """Удаляет пересечения между разными типами: оставляет более длинные, при равной длине — более ранние."""
    order = sorted(spans, key=lambda x: (-(x[1]-x[0]), x[0], x[2]))
    kept: List[Tuple[int,int,str]] = []
    for s, e, k in order:
        if all(e <= ks or s >= ke for (ks, ke, _) in kept):
            kept.append((s, e, k))
    kept.sort(key=lambda x: (x[0], x[1]))
    return kept

def bio_postprocess(
    labels: List[str],
    offsets: List[Tuple[int,int]],
    text: str,
    *,
    trim_whitespace: bool = True,
    merge_same_type: bool = True,
    remove_overlaps: bool = True,
    return_bio_prefix: bool = False,
) -> List[Dict[str, Any]]:
    """
    Склейка BIO и восстановление символьных индексов.
    :return: [{"start_index","end_index","entity"}]
    """
    if len(labels) != len(offsets):
        raise ValueError(f"len(labels) != len(offsets): {len(labels)} vs {len(offsets)}")

    # 1) BIO -> токенные спаны
    tok_spans = _bio_to_token_spans(labels)

    # 2) токенные -> символьные
    char_spans: List[Tuple[int,int,str]] = []
    L = len(text)
    for ti, tj, kind in tok_spans:
        head = _first_valid_offset(offsets, ti, tj)
        tail = _last_valid_offset(offsets, ti, tj)
        if head is None or tail is None:
            continue
        s = max(0, min(head[0], L))
        e = max(0, min(tail[1], L))
        if kind == "TYPE":
            s, e = _shrink_type_by_stopwords(text, s, e)
        if trim_whitespace:
            s, e = _trim_spaces(text, s, e)
        if s < e:
            char_spans.append((s, e, kind))

    # 3) мержим одинаковые типы (на случай разрывов внутри одного вида)
    if merge_same_type:
        char_spans = _merge_same_type_char_spans(char_spans)

    # 4) убираем пересечения между типами
    if remove_overlaps:
        char_spans = _drop_overlaps_by_longest(char_spans)

    # 5) сбор ответа
    out: List[Dict[str, Any]] = []
    for s, e, k in char_spans:
        ent = f"B-{k}" if return_bio_prefix else k
        out.append({"start_index": int(s), "end_index": int(e), "entity": ent})
    return out

# ---- Доп: подтверждение единиц/процентов по regex (использовать в API после bio_postprocess) ----
def merge_units(text: str, spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Добавляет/исправляет VOLUME и PERCENT по regex, не заезжая на существующие спаны других типов.
    """
    keep = [dict(s) for s in spans if s.get("entity") not in {"VOLUME", "PERCENT"}]

    def add_span(s: int, e: int, ent: str):
        for t in keep:
            if not (e <= t["start_index"] or s >= t["end_index"]):
                return  # пересекается — пропускаем
        keep.append({"start_index": s, "end_index": e, "entity": ent})

    for m in RE_VOL.finditer(text):
        add_span(m.start(), m.end(), "VOLUME")
    for m in RE_PCT.finditer(text):
        add_span(m.start(), m.end(), "PERCENT")

    keep.sort(key=lambda x: (x["start_index"], x["end_index"]))
    return keep
