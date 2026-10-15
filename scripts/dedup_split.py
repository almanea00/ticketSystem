"""Remove leakage: drop every train ticket that is a reworded twin of a val or test ticket.

The source file holds several reworded versions of many tickets (its "version" column), and 61% of test
tickets had a twin in train (TF-IDF cosine >= 0.5) with the same team 98% of the time, so a word-counting
classifier scored 67 F1 by memorising twins. Test/val stay unchanged (their Arabic translations stay valid);
train twins get split = "dropped_twin" (also their Arabic translations). Re-run all evaluations after this.

Usage: python scripts/dedup_split.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parent.parent
TWIN = 0.5


def main():
    d = pd.read_parquet(ROOT / "data/tickets.parquet")
    en = d[d["lang"] == "en"]
    text = lambda x: (x["subject"] + ". " + x["body"]).tolist()
    tr, held = en[en["split"] == "train"], en[en["split"].isin(["val", "test"])]
    tf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(text(en))
    S = (tf.transform(text(tr)) @ tf.transform(text(held)).T).tocsr()
    best = np.asarray(S.max(1).todense()).ravel()
    drop = set(tr["id"][best >= TWIN])
    d.loc[d["id"].isin(drop) & (d["split"] == "train"), "split"] = "dropped_twin"
    d.to_parquet(ROOT / "data/tickets.parquet")
    print(f"dropped {len(drop)} train tickets with a val/test twin (cosine >= {TWIN})")
    print(d.groupby(["split", "lang"]).size())


if __name__ == "__main__":
    main()
