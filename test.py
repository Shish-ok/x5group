import json, pandas as pd
rows=[]
with open(r"x5_bio_rubert_nfc\train_bio.val.jsonl","r",encoding="utf-8") as f:
    for ln in f:
        if ln.strip():
            o=json.loads(ln)
            rows.append({"id":o["id"], "sample":o["text"]})
pd.DataFrame(rows).to_csv("tmp_val.csv", index=False)
print("tmp_val.csv written:", len(rows))
