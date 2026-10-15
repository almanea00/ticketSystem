"""Leakage check for the Lab 3.4 benchmark: is any trained model's score explained by near-copies of test messages
in its training data? 1) exact id / text overlap, 2) nearest-train similarity (char 3-5gram TF-IDF) per test
message, 3) macro-F1 / accuracy per method on test messages WITH vs WITHOUT a close training twin.
Usage: python scripts/leak_check.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.bench import data  # noqa: E402

train, val, test, labels = data()
print("id overlap train/test:", len(set(train.id) & set(test.id)), "| val/test:", len(set(val.id) & set(test.id)))
norm = lambda s: s.str.lower().str.replace(r"\s+", " ", regex=True).str.strip()  # noqa: E731
print("exact text overlap train/test:", len(set(norm(train.text)) & set(norm(test.text))))

rows = []
for lang in ("en", "ar", "najdi"):
    tr = train[train.lang == lang] if lang == "en" else train[train.lang == lang]
    te = test[test.lang == lang].reset_index(drop=True)
    v = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit(pd.concat([tr.text, te.text]))
    S = (v.transform(te.text) @ v.transform(tr.text).T).tocsr()
    best = np.asarray(S.max(1).todense()).ravel()
    te["nearest"] = best
    rows.append(te)
    print(f"{lang}: nearest-train similarity median {np.median(best):.2f}, share >= 0.8: {100*(best>=0.8).mean():.1f}%, >= 0.9: {100*(best>=0.9).mean():.1f}%")
te = pd.concat(rows, ignore_index=True)
cut = 0.8
out = []
for m in ["tfidf_lr", "e5_lr", "distilmbert_ft", "marbert_ft", "qwen05_lora", "embgemma_lr", "d1_zeroshot", "laya_ml_zeroshot"]:
    p = pd.read_parquet(ROOT / f"results/bench/{m}.parquet")
    j = te.merge(p[["id", "lang", "pred"]], on=["id", "lang"])
    for lang, g in j.groupby("lang"):
        hi, lo = g[g.nearest >= cut], g[g.nearest < cut]
        out.append({"method": m, "lang": lang, f"acc, has twin (>= {cut})": 100 * (hi.pred == hi.intent).mean(),
                    f"acc, no twin (< {cut})": 100 * (lo.pred == lo.intent).mean(), "n twin": len(hi), "n no twin": len(lo)})
r = pd.DataFrame(out)
r["gap"] = r.iloc[:, 2] - r.iloc[:, 3]
print(r.round(1).to_markdown(index=False))
