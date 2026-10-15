"""Second dataset: Bitext customer support (CDLA-Sharing-1.0), 26.9k short messages with clean, balanced labels:
11 categories (ACCOUNT, ORDER, REFUND, ...) and 27 intents (cancel_order, track_refund, ...), plus agent responses.
Used to check whether low scores on the Tobi-Bueck tickets come from the models or from that dataset's labels.

  prep   split 80/10/10 stratified by intent, drop train twins of val/test (TF-IDF cosine >= 0.5, same rule as
         Tobi-Bueck), translate the test set to فصحى and Najdi (NLLB-200) -> data/bitext.parquet
  eval   category + intent: majority, TF-IDF+LR, EmbeddingGemma+LR, d1 zero-shot, Laya zero-shot
         -> results/bitext_classify.csv

Usage: python scripts/bitext.py prep | eval [--device cuda:0] [--laya "laya-ara,laya (en)"]
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "data/bitext.parquet"
TWIN = 0.5


def nice(label: str) -> str:
    return label.replace("_", " ").capitalize()


def prep():
    from datasets import load_dataset
    from sklearn.feature_extraction.text import TfidfVectorizer
    from scripts.prep_data import Translator
    d = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train").to_pandas()
    d = d.rename(columns={"instruction": "text", "response": "answer"})
    d = d.drop_duplicates("text").reset_index(drop=True)
    d["id"] = [f"B{i:05d}" for i in range(len(d))]
    rng = np.random.default_rng(0)
    d["split"] = "train"
    for _, g in d.groupby("intent"):
        idx = rng.permutation(g.index.to_numpy())
        n = len(idx) // 10
        d.loc[idx[:n], "split"] = "test"
        d.loc[idx[n:2 * n], "split"] = "val"
    tf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(d["text"])
    tr, held = d[d["split"] == "train"], d[d["split"] != "train"]
    S = (tf.transform(tr["text"]) @ tf.transform(held["text"]).T).tocsr()
    best = np.asarray(S.max(1).todense()).ravel()
    d.loc[tr.index[best >= TWIN], "split"] = "dropped_twin"
    print(f"dropped {(best >= TWIN).sum()} train twins of val/test", d["split"].value_counts().to_dict(), flush=True)
    d["lang"] = "en"
    t = Translator()
    test = d[d["split"] == "test"]
    parts = [d]
    for code, lang in (("arb_Arab", "ar"), ("ars_Arab", "najdi")):
        x = test.copy()
        x["text"] = t.texts(test["text"].tolist(), code)
        x["lang"] = lang
        parts.append(x)
        print(lang, list(zip(test["text"].head(6), x["text"].head(6))), flush=True)
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(OUT)
    print(out.groupby(["split", "lang"]).size())


def evaluate(args):
    import torch
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from scripts.eval_classify import embed
    d = pd.read_parquet(OUT)
    train = d[(d["split"] == "train") & (d["lang"] == "en")].reset_index(drop=True)
    test = d[d["split"] == "test"].reset_index(drop=True)
    labels = {"category": sorted(d["category"].unique()), "intent": sorted(d["intent"].unique())}
    preds = {}
    for task in labels:
        preds[("majority", task)] = [train[task].value_counts().index[0]] * len(test)
    tf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(train["text"])
    for task in labels:
        lr = LogisticRegression(max_iter=2000, C=5).fit(tf.transform(train["text"]), train[task])
        preds[("tfidf+LR", task)] = lr.predict(tf.transform(test["text"]))
    E = embed(train["text"].tolist() + test["text"].tolist(), args.device)
    for task in labels:
        lr = LogisticRegression(max_iter=3000, C=10).fit(E[:len(train)], train[task])
        preds[("emb+LR", task)] = lr.predict(E[len(train):])
    print("baselines done", flush=True)

    # zero-shot decision models: the same two questions for d1 and Laya
    questions = {"category": {"type": "choice", "instructions": "What is this customer message about?",
                              "criteria": {c: nice(c) for c in labels["category"]}},
                 "intent": {"type": "choice", "instructions": "What does the customer want to do?",
                            "criteria": {i: nice(i) for i in labels["intent"]}}}
    states = [{"message": t} for t in test["text"]]
    speed = {}
    from ticketassist.decide import MODEL as D1
    from transformers import AutoModel
    m = AutoModel.from_pretrained(D1, trust_remote_code=True, dtype=torch.bfloat16, device_map={"": args.device}).eval()
    t = time.time()
    out = []
    with torch.no_grad():
        for s in range(0, len(states), 32):
            out += m.system_one_batch([(st, questions) for st in states[s:s + 32]])
    speed["d1 zero-shot"] = 1000 * (time.time() - t) / len(states)
    for task in labels:
        preds[("d1 zero-shot", task)] = [o["answers"][task]["choice"] for o in out]
    del m
    torch.cuda.empty_cache()
    print("d1 done", flush=True)

    os.environ.setdefault("USE_TF", "0")
    import laya
    from scripts.eval_laya import MODELS
    for name in [n for n in args.laya.split(",") if n]:
        repo, sub = MODELS[name]
        agent = laya.load(repo, device=args.device, subfolder=sub)
        t = time.time()
        out = []
        for s in range(0, len(states), 32):
            out += agent.predict_batch(states[s:s + 32], questions)
        speed[name + " zero-shot"] = 1000 * (time.time() - t) / len(states)
        for task in labels:
            preds[(name + " zero-shot", task)] = [o["answers"][task]["choice"] for o in out]
        del agent
        torch.cuda.empty_cache()
        print(name, "done", flush=True)

    rows = []
    for (method, task), p in preds.items():
        for lang in ("en", "ar", "najdi"):
            msk = (test["lang"] == lang).to_numpy()
            y, yp = test.loc[msk, task], np.asarray(p)[msk]
            rows.append({"method": method, "task": task, "lang": lang, "n": int(msk.sum()),
                         "macro-F1": 100 * f1_score(y, yp, average="macro", labels=labels[task], zero_division=0),
                         "accuracy": 100 * accuracy_score(y, yp), "ms/ticket": speed.get(method)})
    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "results/bitext_classify.csv", index=False)
    for task in labels:
        t = res[res["task"] == task]
        print(f"\n{task} ({len(labels[task])} classes): macro-F1 (accuracy)")
        print(t.assign(v=t["macro-F1"].round(1).astype(str) + " (" + t["accuracy"].round(1).astype(str) + ")")
              .pivot(index="method", columns="lang", values="v")[["en", "ar", "najdi"]].to_markdown())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prep", "eval"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--laya", default="laya-ara,laya (en),laya multilingual")
    a = ap.parse_args()
    prep() if a.step == "prep" else evaluate(a)
