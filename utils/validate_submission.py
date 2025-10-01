"""
validate_submission.py — проверка submission.csv (и опционально фиксация).

Проверяет:
- наличие колонок id/annotation;
- парсинг annotation (list of (start,end,tag));
- валидность типов (TYPE/BRAND/VOLUME/PERCENT);
- start<end, индексы не отрицательные;
- (если дан test.csv) end <= len(text) после NFC-нормализации;
- отсутствие пересечений, сортировку спанов;
- покрытие id: есть ли id вне test.csv и пропавшие id из test.csv.

Опционально: --fix-out сохраняет исправленную версию submission (сортировка, удаление/обрезка битых, снятие пересечений).
"""

import argparse, ast, csv, io, os, sys
from typing import Any, Dict, List, Tuple, Optional
import pandas as pd

VALID_TYPES = {"TYPE","BRAND","VOLUME","PERCENT"}

# --- NFC normalize (как в остальных скриптах) ---
try:
    from utils.textnorm import normalize_nfc
except Exception:
    import unicodedata
    def normalize_nfc(s: str) -> str:
        return unicodedata.normalize("NFC", str(s))

def sniff_sep(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            sample = f.read(4096)
        dialect = csv.Sniffer().sniff(sample, delimiters=";,|\t,")
        return dialect.delimiter
    except Exception:
        return None

def parse_ann(cell: Any) -> List[Tuple[int,int,str]]:
    if cell is None:
        return []
    s = str(cell).strip()
    if not s or s == "nan":
        return []
    try:
        obj = ast.literal_eval(s)
        out = []
        for item in obj:
            if not (isinstance(item, (list, tuple)) and len(item) == 3):
                continue
            a, b, t = item
            t = str(t)
            if "-" in t:  # 'B-TYPE' -> 'TYPE'
                t = t.split("-",1)[-1]
            out.append((int(a), int(b), t))
        return out
    except Exception:
        return []

def drop_overlaps_keep_longer(spans: List[Tuple[int,int,str]]) -> List[Tuple[int,int,str]]:
    # сортируем по длине убыв., затем по началу
    order = sorted(spans, key=lambda x: (-(x[1]-x[0]), x[0], x[2]))
    kept: List[Tuple[int,int,str]] = []
    for s,e,k in order:
        ok = True
        for ks,ke,kk in kept:
            if not (e <= ks or s >= ke):
                ok = False
                break
        if ok:
            kept.append((s,e,k))
    kept.sort(key=lambda x: (x[0], x[1], x[2]))
    return kept

def validate(sub_csv: str, test_csv: Optional[str], id_col: str, text_col: str,
             show_errors: int = 20, fix_out: Optional[str] = None) -> None:
    sub_sep = sniff_sep(sub_csv)
    df = pd.read_csv(sub_csv, sep=sub_sep if sub_sep else None, engine="python")
    if id_col not in df.columns or "annotation" not in df.columns:
        raise SystemExit(f"[ERR] В {sub_csv} нет необходимых колонок: {id_col} и annotation.")

    texts: Dict[Any, str] = {}
    text_len: Dict[Any, int] = {}
    if test_csv and os.path.isfile(test_csv):
        test_sep = sniff_sep(test_csv)
        dt = pd.read_csv(test_csv, sep=test_sep if test_sep else None, engine="python")
        if text_col not in dt.columns:
            print(f"[WARN] В {test_csv} нет колонки {text_col}; пропускаю проверку длины.", file=sys.stderr)
        else:
            if id_col in dt.columns:
                for _, r in dt[[id_col, text_col]].iterrows():
                    s = normalize_nfc(str(r[text_col]))
                    texts[r[id_col]] = s
                    text_len[r[id_col]] = len(s)
            else:
                # если id нет — валидируем только по длине текста по индексам
                for idx, s in enumerate(dt[text_col].astype(str).tolist()):
                    s = normalize_nfc(s)
                    texts[idx] = s
                    text_len[idx] = len(s)

    n = len(df)
    err_parse = 0
    err_type = 0
    err_order = 0
    err_overlap = 0
    err_bounds = 0
    empty_ok = 0

    examples = []

    cleaned_rows = []

    for _, row in df.iterrows():
        rid = row[id_col]
        anns = parse_ann(row["annotation"])
        if row["annotation"] and not anns and str(row["annotation"]).strip():
            err_parse += 1
            if len(examples) < show_errors:
                examples.append((rid, "parse_error", row["annotation"]))
            # даже если не распарсили — продолжим
        # фильтруем по типам и базовой валидности
        valid = []
        for (s,e,t) in anns:
            if t not in VALID_TYPES:
                err_type += 1
                continue
            if s < 0 or e < 0 or s >= e:
                err_bounds += 1
                continue
            # если есть текст — проверяем верхнюю границу
            if rid in text_len and e > text_len[rid]:
                err_bounds += 1
                # можно поджать до границ: e = min(e, text_len[rid])
                # пропустим валидацию, а фиксер (ниже) подожмёт при необходимости
            valid.append((s,e,t))

        if not valid:
            empty_ok += 1

        # проверка сортировки и пересечений
        sorted_valid = sorted(valid, key=lambda x: (x[0], x[1], x[2]))
        if sorted_valid != valid:
            err_order += 1
        # пересечения
        overlap = False
        last_e = -1
        for (s,e,t) in sorted_valid:
            if s < last_e:
                overlap = True
                break
            last_e = e
        if overlap:
            err_overlap += 1

        # фиксация (если нужно)
        if fix_out:
            L = text_len.get(rid, None)
            fixed = []
            for (s,e,t) in sorted_valid:
                if L is not None:
                    s = max(0, min(s, L))
                    e = max(0, min(e, L))
                if s < e:
                    fixed.append((s,e,t))
            fixed = drop_overlaps_keep_longer(fixed)
            cleaned_rows.append((rid, str(fixed)))

    # покрытие id при наличии test.csv
    missing_ids = set()
    extra_ids = set()
    if texts:
        test_ids = set(texts.keys())
        sub_ids = set(df[id_col].tolist())
        missing_ids = test_ids - sub_ids
        extra_ids   = sub_ids - test_ids

    # вывод
    print("=== Submission check ===")
    print(f"rows: {n}")
    if texts:
        print(f"test rows: {len(texts)}  | missing ids: {len(missing_ids)}  | extra ids: {len(extra_ids)}")
    print(f"parse errors: {err_parse}")
    print(f"invalid types: {err_type}")
    print(f"invalid bounds/start>=end: {err_bounds}")
    print(f"unsorted rows: {err_order}")
    print(f"overlaps: {err_overlap}")
    print(f"empty annotations: {empty_ok}")

    if examples:
        print("\n--- examples ---")
        for rid, etype, payload in examples[:show_errors]:
            print(f"id={rid}  {etype}: {payload}")

    if fix_out:
        # сохраним исправленный файл
        out_path = fix_out
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        out_df = pd.DataFrame(cleaned_rows, columns=[id_col, "annotation"])
        out_df.to_csv(out_path, index=False)
        print(f"\n[fix] cleaned submission saved: {out_path} (rows={len(out_df)})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True, help="submission.csv (id, annotation)")
    ap.add_argument("--test-csv", default=None, help="(опц.) test.csv с текстами для проверки границ")
    ap.add_argument("--id-col", default="id")
    ap.add_argument("--text-col", default="sample")
    ap.add_argument("--show-errors", type=int, default=20)
    ap.add_argument("--fix-out", default=None, help="(опц.) путь для сохранения исправленной версии")
    args = ap.parse_args()

    validate(args.submission, args.test_csv, args.id_col, args.text_col,
             show_errors=args.show_errors, fix_out=args.fix_out)

if __name__ == "__main__":
    main()
