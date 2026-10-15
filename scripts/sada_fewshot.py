"""d1-3B few-shot Saudi dialect ID on SADA: k labelled SADA training sentences per dialect inside the state, no training.
Same 1,539 test transcripts (513 per dialect) as scripts/sada_dialect.py. k in {0, 1, 3, 5}; the examples are fixed
(seed 0) and shuffled, so every test sentence sees the same set.
Writes results/sada_fewshot.csv. Usage: python scripts/sada_fewshot.py [--device cuda:0]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.sada_dialect import OPTS, load  # noqa: E402
from scripts.policy_bench import load_d1  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cuda:0")
args = ap.parse_args()
test = load("test", 600)
pool = load("train", 50, seed=1)
m = load_d1(args.device)
labels = list(OPTS)
rows = []
for k in (0, 1, 3, 5):
    ex = pool.groupby("dialect").head(k).sample(frac=1, random_state=0)
    shots = [{"text": t, "dialect": d} for t, d in zip(ex.text, ex.dialect)]
    instr = ("Which Saudi Arabic dialect is the text written in?" +
             (" Use the labelled examples as reference." if k else ""))
    q = {"d": {"type": "choice", "instructions": instr, "criteria": OPTS}}
    pred = []
    with torch.inference_mode():
        for s in range(0, len(test), 16):
            states = [({"labelled_examples": shots, "text": t} if k else {"text": t}) for t in test.text[s:s + 16]]
            pred += [r["answers"]["d"]["choice"] for r in m.system_one_batch([(st, q) for st in states])]
    y = test.dialect.tolist()
    rows.append({"shots per dialect": k, "total shots": 3 * k, "macro-F1": 100 * f1_score(y, pred, average="macro"),
                 "accuracy": 100 * accuracy_score(y, pred), **{f"F1 {l}": 100 * f1_score(y, pred, labels=[l], average="macro") for l in labels}})
    print(rows[-1], flush=True)
    print(pd.DataFrame(confusion_matrix(y, pred, labels=labels), index=[f"true {l}" for l in labels], columns=[f"pred {l}" for l in labels]), flush=True)
pd.DataFrame(rows).to_csv(ROOT / "results/sada_fewshot.csv", index=False)
print(pd.DataFrame(rows).round(1).to_markdown(index=False))
