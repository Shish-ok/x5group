"""
metrics_bio.py — метрики NER по BIO

Функции:
  • Entity-level: строгий матч (тип + точный токенный span)
  • По классам: P / R / F1 + support; macro-F1; micro-F1
  • (Опц.) Token-level сводка

Форматы входа:
  GOLD (JSONL): [{"id", "tokens", "labels", "offsets"}, ...]
  PRED (JSONL):
    A) [{"id", "labels"}]  — список BIO-меток на тех же токенах
    B) [{"id", "spans":[{"start_index","end_index","entity"}, ...]}]
       В этом случае конвертируем spans → BIO по offsets из gold.

Примеры:
  python metrics_bio.py --gold x5_bio/train_bio.val.jsonl --pred runs/pred.val.jsonl
  python metrics_bio.py --gold x5_bio/train_bio.val.jsonl --pred runs/pred_spans.val.jsonl --token-level
"""

from __future__ import annotations
import argparse, io, json, sys
from typing import Dict, List, Tuple, Iterable, Any, Set, Optional
from collections import defaultdict, Counter

VALID_KINDS: Set[str] = {"TYPE","BRAND","VOLUME","PERCENT"}

# ---------- I/O ----------
def read_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    with io.open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def index_by_id(rows: Iterable[dict]) -> Dict[Any, dict]:
    d: Dict[Any, dict] = {}
    for r in rows:
        if "id" not in r:
            raise ValueError("Каждая строка должна иметь поле 'id'.")
        d[r["id"]] = r
    return d

# ---------- Helpers (type-safe) ----------
def ensure_list_str(obj: Any, *, field: str, rid: Any) -> List[str]:
    if not isinstance(obj, list):
        raise ValueError(f"[id={rid}] Ожидал list в поле '{field}', получено {type(obj).__name__}")
    try:
        return [str(x) for x in obj]
    except Exception:
        raise ValueError(f"[id={rid}] Не удалось привести элементы '{field}' к строкам.")

def ensure_offsets(obj: Any, *, field: str, rid: Any) -> List[Tuple[int,int]]:
    if not isinstance(obj, list):
        raise ValueError(f"[id={rid}] Ожидал list в поле '{field}'.")
    out: List[Tuple[int,int]] = []
    for it in obj:
        if (not isinstance(it, (list, tuple))) or len(it) != 2:
            raise ValueError(f"[id={rid}] Ожидал пары (start,end) в '{field}'.")
        s, e = it
        try:
            s_i = int(s); e_i = int(e)
        except Exception:
            raise ValueError(f"[id={rid}] Нельзя преобразовать offsets к int: {it}")
        out.append((s_i, e_i))
    return out

def extract_gold_labels(row: dict) -> List[str]:
    rid = row.get("id")
    if "labels" not in row:
        raise ValueError(f"[gold id={rid}] Нет поля 'labels'.")
    labels = ensure_list_str(row["labels"], field="labels", rid=rid)
    return labels

def extract_pred_labels_or_spans(pred_row: dict, gold_offsets: List[Tuple[int,int]]) -> List[str]:
    rid = pred_row.get("id")
    # Случай A: labels
    if "labels" in pred_row:
        return ensure_list_str(pred_row["labels"], field="labels", rid=rid)
    # Случай B: spans
    if "spans" not in pred_row:
        raise ValueError(f"[pred id={rid}] Нет ни 'labels', ни 'spans'.")
    spans_obj = pred_row["spans"]
    if not isinstance(spans_obj, list):
        raise ValueError(f"[pred id={rid}] Поле 'spans' должно быть списком.")
    return spans_to_token_bio(gold_offsets, spans_obj, rid=rid)

# ---------- BIO utils ----------
def bio_to_entities(labels: List[str]) -> List[Tuple[int,int,str]]:
    """
    BIO → список (start_tok, end_tok_exclusive, kind).
    Некорректные 'I' без 'B' начинаем как новую сущность.
    """
    ents: List[Tuple[int,int,str]] = []
    start: Optional[int] = None
    cur_kind: Optional[str] = None
    for i, lab in enumerate(labels):
        if lab == "O" or lab is None:
            if start is not None and cur_kind is not None:
                ents.append((start, i, cur_kind))
            start, cur_kind = None, None
            continue
        if "-" in lab:
            prefix, kind = lab.split("-", 1)
        else:
            prefix, kind = "B", lab
        if prefix == "B":
            if start is not None and cur_kind is not None:
                ents.append((start, i, cur_kind))
            start, cur_kind = i, kind
        elif prefix == "I":
            if start is None or cur_kind != kind:
                if start is not None and cur_kind is not None:
                    ents.append((start, i, cur_kind))
                start, cur_kind = i, kind
        else:
            if start is not None and cur_kind is not None:
                ents.append((start, i, cur_kind))
            start, cur_kind = None, None
    if start is not None and cur_kind is not None:
        ents.append((start, len(labels), cur_kind))
    return ents

def spans_to_token_bio(offsets: List[Tuple[int,int]], spans: List[dict], *, rid: Any) -> List[str]:
    """
    Символьные спаны → BIO по токенам (строгий охват пересечения).
    spans: [{"start_index":..., "end_index":..., "entity":"B-XXX"/"I-XXX"/"XXX"}, ...]
    """
    # Нормализуем к (s,e,kind)
    norm: List[Tuple[int,int,str]] = []
    for i, sp in enumerate(spans):
        if not isinstance(sp, dict):
            raise ValueError(f"[pred id={rid}] Элемент spans[{i}] не dict.")
        s_raw = sp.get("start_index", None)
        e_raw = sp.get("end_index", None)
        ent_raw = sp.get("entity", None)
        if s_raw is None or e_raw is None or ent_raw is None:
            # пропускаем неполные элементы — не валим весь батч
            continue
        try:
            s = int(s_raw); e = int(e_raw)
        except Exception:
            # пропускаем неконвертируемые
            continue
        ent = str(ent_raw)
        kind = ent.split("-", 1)[1] if "-" in ent else ent
        if kind in VALID_KINDS and s < e:
            norm.append((s, e, kind))
    # Склеим пересекающиеся одного вида
    norm.sort(key=lambda x: (x[0], x[1]))
    merged: List[Tuple[int,int,str]] = []
    if norm:
        cs, ce, ck = norm[0]
        for s, e, k in norm[1:]:
            if k == ck and s <= ce:
                ce = max(ce, e)
            else:
                merged.append((cs, ce, ck))
                cs, ce, ck = s, e, k
        merged.append((cs, ce, ck))
    # Проекция на токены
    labels = ["O"] * len(offsets)
    for s, e, k in merged:
        first = True
        for i, (ts, te) in enumerate(offsets):
            if te <= s or ts >= e:
                continue
            labels[i] = ("B-" if first else "I-") + k
            first = False
    return labels

# ---------- Scoring ----------
def prf(tp: int, fp: int, fn: int) -> Tuple[float,float,float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1

def score_entity_level(gold_rows: List[dict], pred_rows: List[dict]) -> Dict[str,Any]:
    gold = index_by_id(gold_rows)
    pred = index_by_id(pred_rows)
    ids = sorted(set(gold.keys()) & set(pred.keys()))
    if not ids:
        raise ValueError("Нет пересечения по 'id' между gold и pred.")

    TP: Counter[str] = Counter(); FP: Counter[str] = Counter(); FN: Counter[str] = Counter()
    SUPPORT: Counter[str] = Counter()

    for rid in ids:
        g = gold[rid]; p = pred[rid]
        g_labels = extract_gold_labels(g)
        g_offsets = ensure_offsets(g.get("offsets"), field="offsets", rid=rid)

        p_labels = extract_pred_labels_or_spans(p, g_offsets)

        # BIO → сущности
        g_ents = bio_to_entities(g_labels)
        p_ents = bio_to_entities(p_labels)

        # support по gold
        for _, _, k in g_ents:
            if k in VALID_KINDS:
                SUPPORT[k] += 1

        g_set = {(s, e, k) for (s, e, k) in g_ents if k in VALID_KINDS}
        p_set = {(s, e, k) for (s, e, k) in p_ents if k in VALID_KINDS}

        for s, e, k in p_set:
            if (s, e, k) in g_set:
                TP[k] += 1
            else:
                FP[k] += 1
        for s, e, k in g_set:
            if (s, e, k) not in p_set:
                FN[k] += 1

    per_class: Dict[str, Dict[str, Any]] = {}
    sum_tp = sum_fp = sum_fn = 0
    for k in sorted(VALID_KINDS):
        tp, fp, fn = TP[k], FP[k], FN[k]
        p, r, f1 = prf(tp, fp, fn)
        per_class[k] = {"precision": p, "recall": r, "f1": f1, "support": SUPPORT[k], "tp": tp, "fp": fp, "fn": fn}
        sum_tp += tp; sum_fp += fp; sum_fn += fn
    macro_f1 = sum(v["f1"] for v in per_class.values()) / len(VALID_KINDS)
    micro_p, micro_r, micro_f1 = prf(sum_tp, sum_fp, sum_fn)

    return {
        "per_class": per_class,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "micro_p": micro_p,
        "micro_r": micro_r,
        "support_total": sum(SUPPORT.values()),
        "tp_total": sum_tp, "fp_total": sum_fp, "fn_total": sum_fn,
        "evaluated_ids": len(ids),
    }

def score_token_level(gold_rows: List[dict], pred_rows: List[dict]) -> Dict[str,Any]:
    gold = index_by_id(gold_rows)
    pred = index_by_id(pred_rows)
    ids = sorted(set(gold.keys()) & set(pred.keys()))
    TP: Counter[str] = Counter(); FP: Counter[str] = Counter(); FN: Counter[str] = Counter()
    SUPPORT: Counter[str] = Counter()

    for rid in ids:
        g = gold[rid]; p = pred[rid]
        g_labels = extract_gold_labels(g)
        g_offsets = ensure_offsets(g.get("offsets"), field="offsets", rid=rid)

        p_labels = extract_pred_labels_or_spans(p, g_offsets)

        if len(g_labels) != len(p_labels):
            raise ValueError(f"[id={rid}] Длины не совпадают: gold={len(g_labels)} pred={len(p_labels)}")

        for gl, pl in zip(g_labels, p_labels):
            if gl == "O":
                if pl != "O":
                    pk = pl.split("-", 1)[1] if "-" in pl else pl
                    if pk in VALID_KINDS:
                        FP[pk] += 1
                continue

            gk = gl.split("-", 1)[1] if "-" in gl else gl
            SUPPORT[gk] += 1
            if pl == "O":
                FN[gk] += 1
            else:
                pk = pl.split("-", 1)[1] if "-" in pl else pl
                if pk == gk:
                    TP[gk] += 1
                else:
                    FN[gk] += 1; FP[pk] += 1

    per_class: Dict[str, Dict[str, Any]] = {}
    sum_tp = sum_fp = sum_fn = 0
    for k in sorted(VALID_KINDS):
        tp, fp, fn = TP[k], FP[k], FN[k]
        p, r, f1 = prf(tp, fp, fn)
        per_class[k] = {"precision": p, "recall": r, "f1": f1, "support_tokens": SUPPORT[k], "tp": tp, "fp": fp, "fn": fn}
        sum_tp += tp; sum_fp += fp; sum_fn += fn
    macro_f1 = sum(v["f1"] for v in per_class.values()) / len(VALID_KINDS)
    micro_p, micro_r, micro_f1 = prf(sum_tp, sum_fp, sum_fn)

    return {"per_class": per_class, "macro_f1": macro_f1, "micro_f1": micro_f1, "micro_p": micro_p, "micro_r": micro_r}

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="NER metrics (BIO): entity-level и опц. token-level")
    ap.add_argument("--gold", required=True, help="JSONL с эталонной BIO-разметкой")
    ap.add_argument("--pred", required=True, help="JSONL с предсказаниями (labels или spans)")
    ap.add_argument("--token-level", action="store_true", help="Дополнительно посчитать токен-уровень")
    ap.add_argument("--dump-json", default="", help="Путь для сохранения метрик в JSON (опционально)")
    args = ap.parse_args()

    gold = read_jsonl(args.gold)
    pred = read_jsonl(args.pred)

    ent = score_entity_level(gold, pred)
    print("=== Entity-level (strict) ===")
    print(f"evaluated_ids: {ent['evaluated_ids']}")
    print(f"support_total: {ent['support_total']}")
    print("class\tprecision\trecall\tf1\tsupport\t(tp/fp/fn)")
    for k in sorted(ent["per_class"].keys()):
        v = ent["per_class"][k]
        print(f"{k}\t{v['precision']:.4f}\t{v['recall']:.4f}\t{v['f1']:.4f}\t{v['support']}\t({v['tp']}/{v['fp']}/{v['fn']})")
    print(f"macro_f1\t{ent['macro_f1']:.4f}")
    print(f"micro_p\t{ent['micro_p']:.4f}")
    print(f"micro_r\t{ent['micro_r']:.4f}")
    print(f"micro_f1\t{ent['micro_f1']:.4f}")

    out: Dict[str, Any] = {"entity_level": ent}

    if args.token_level:
        tok = score_token_level(gold, pred)
        print("\n=== Token-level (by label) ===")
        print("class\tprecision\trecall\tf1\tsupport_tokens")
        for k in sorted(tok["per_class"].keys()):
            v = tok["per_class"][k]
            print(f"{k}\t{v['precision']:.4f}\t{v['recall']:.4f}\t{v['f1']:.4f}\t{v['support_tokens']}")
        print(f"macro_f1\t{tok['macro_f1']:.4f}")
        out["token_level"] = tok

    if args.dump_json:
        with io.open(args.dump_json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
