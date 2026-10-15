"""Stricter leakage audit for e5 + LR (and every trained method that uses the same split).
 1) cross-language: for every test message id, are ALL its language versions in test only (never in train/val)?
 2) strict re-split: drop every train message (any language) whose character 3-5-gram TF-IDF cosine to ANY test
    message (any language) is >= 0.6, retrain e5 + LR on what is left, compare macro-F1 with the normal split.
Usage: python scripts/leak_check_strict.py [--device cuda:1]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.bench import data, extra_sets  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cuda:1")
ap.add_argument("--cut", type=float, default=0.6)
args = ap.parse_args()
train, val, test, labels = data()
raw = pd.read_parquet(ROOT / "data/bitext.parquet")
ar_all = pd.read_parquet(ROOT / "data/bitext_train_ar.parquet")
ids_test = set(test.id)
where = pd.concat([raw[["id", "split", "lang"]], ar_all[["id", "split", "lang"]]])
bad = where[where.id.isin(ids_test) & (where.split != "test")]
print("1) test ids that also appear outside the test split, in any language:", len(bad))
print("   train ids:", train.id.nunique(), "| test ids:", test.id.nunique(), "| overlap:", len(set(train.id) & ids_test))

v = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2).fit(pd.concat([train.text, test.text]))
S = (v.transform(train.text) @ v.transform(test.text).T).tocsr()
best = np.asarray(S.max(1).todense()).ravel()
keep = best < args.cut
print(f"2) train messages with a test message at cosine >= {args.cut} (any language): {(~keep).sum()} of {len(train)} dropped",
      train[~keep].groupby('lang').size().to_dict())
strict = train[keep]
print("   strict train per intent: min", strict.intent.value_counts().min(), "max", strict.intent.value_counts().max())

m = SentenceTransformer("intfloat/multilingual-e5-base", device=args.device)
enc = lambda t: m.encode(["query: " + x for x in t], batch_size=64, normalize_embeddings=True)  # noqa: E731
Etr, Ete = enc(train.text.tolist()), enc(test.text.tolist())
ex = extra_sets()
course = ex[ex.lang.str.startswith("course_")]
Ec = enc(course.text.tolist())
rows = []
for name, mask in (("normal split (10,794)", np.ones(len(train), bool)), (f"strict split ({keep.sum():,})", keep)):
    lr = LogisticRegression(max_iter=3000, C=10).fit(Etr[mask], train.intent[mask])
    p = lr.predict(Ete)
    r = {"e5 + LR trained on": name}
    for lang in ("en", "ar", "najdi"):
        msk = (test.lang == lang).to_numpy()
        r[lang] = round(100 * f1_score(test.intent[msk], p[msk], average="macro"), 1)
    r["course human AR"] = round(100 * f1_score(course.intent, lr.predict(Ec), average="macro"), 1)
    rows.append(r)
print(pd.DataFrame(rows).to_markdown(index=False))
