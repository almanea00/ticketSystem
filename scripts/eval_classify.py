"""Team (queue) and priority on the test tickets (English, فصحى, Najdi): which model decides best?

  majority     always the most common class (floor)
  tfidf+LR     TF-IDF words + logistic regression, trained on the 13k English train tickets
  emb+LR       EmbeddingGemma vectors ("task: classification") + logistic regression, same training data
  d1 zero-shot LiquidAI d1-3B, only the team names/descriptions, no training

Writes results/classify.csv (macro-F1, accuracy per method/task/language), results/classify_preds.parquet,
and confusion matrices for the best method.

Usage: python scripts/eval_classify.py [--limit N]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ticketassist.decide import PRIORITIES, QUEUES, Decider  # noqa: E402
from ticketassist.embed import ticket_text  # noqa: E402

TASKS = {"queue": list(QUEUES), "priority": PRIORITIES}


def embed(texts, device):
    import torch
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("google/embeddinggemma-2", device=device, model_kwargs={"torch_dtype": torch.bfloat16},
                            config_kwargs={"vision_config": None, "audio_config": None})
    m.max_seq_length = 512
    v = m.encode(texts, prompt="task: classification | query: ", batch_size=64, normalize_embeddings=True,
                 show_progress_bar=True)
    del m
    torch.cuda.empty_cache()
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="test tickets per language (0 = all)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    d = pd.read_parquet(ROOT / "data/tickets.parquet")
    train = d[(d["split"] == "train") & (d["lang"] == "en")].reset_index(drop=True)
    test = d[d["split"] == "test"].reset_index(drop=True)
    if args.limit:
        keep = set(test[test["lang"] == "en"]["id"].sample(args.limit, random_state=0))
        test = test[test["id"].isin(keep)].reset_index(drop=True)
    print("train", len(train), "| test", test.groupby("lang").size().to_dict(), flush=True)
    tr_text = [ticket_text(s, b) for s, b in zip(train["subject"], train["body"])]
    te_text = [ticket_text(s, b) for s, b in zip(test["subject"], test["body"])]

    preds = {}
    for task, labels in TASKS.items():
        major = train[task].value_counts().index[0]
        preds[("majority", task)] = [major] * len(test)
    tf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(tr_text)
    Xtr, Xte = tf.transform(tr_text), tf.transform(te_text)
    for task in TASKS:
        lr = LogisticRegression(max_iter=2000, C=5, class_weight="balanced").fit(Xtr, train[task])
        preds[("tfidf+LR", task)] = lr.predict(Xte)
    print("tf-idf done", flush=True)

    E = embed(tr_text + te_text, args.device)
    Etr, Ete = E[:len(tr_text)], E[len(tr_text):]
    for task in TASKS:
        lr = LogisticRegression(max_iter=3000, C=10, class_weight="balanced").fit(Etr, train[task])
        preds[("emb+LR", task)] = lr.predict(Ete)
    print("embedding LR done", flush=True)

    dec = Decider(args.device)
    tri = dec.triage(list(zip(test["subject"], test["body"])))
    preds[("d1 zero-shot", "queue")] = [t["queue"] for t in tri]
    preds[("d1 zero-shot", "priority")] = [t["priority"] for t in tri]
    test["d1_queue_conf"] = [t["queue_conf"] for t in tri]
    test["d1_priority_level"] = [t["priority_level"] for t in tri]
    test["d1_escalate"] = [t["escalate"] for t in tri]

    rows = []
    for (method, task), p in preds.items():
        test[f"{method}|{task}"] = list(p)
        for lang in ("en", "ar", "najdi"):
            m = (test["lang"] == lang).to_numpy()
            if not m.any():
                continue
            y, yp = test.loc[m, task], np.asarray(p)[m]
            rows.append({"method": method, "task": task, "lang": lang, "n": int(m.sum()),
                         "macro-F1": 100 * f1_score(y, yp, average="macro", labels=TASKS[task], zero_division=0),
                         "accuracy": 100 * accuracy_score(y, yp)})
    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "results/classify.csv", index=False)
    test.drop(columns=["answer"]).to_parquet(ROOT / "results/classify_preds.parquet")
    for task in TASKS:
        print(f"\n{task}: macro-F1 (accuracy)")
        t = res[res["task"] == task]
        print(t.assign(v=t["macro-F1"].round(1).astype(str) + " (" + t["accuracy"].round(1).astype(str) + ")")
              .pivot(index="method", columns="lang", values="v").to_markdown())
        best = t[t["lang"] == "en"].sort_values("macro-F1").iloc[-1]["method"]
        en = test["lang"] == "en"
        cm = confusion_matrix(test.loc[en, task], test.loc[en, f"{best}|{task}"], labels=TASKS[task])
        pd.DataFrame(cm, index=TASKS[task], columns=TASKS[task]).to_csv(ROOT / f"results/confusion_{task}.csv")


if __name__ == "__main__":
    main()
