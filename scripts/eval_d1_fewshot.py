"""d1-3B few-shot on the intent benchmark: k English example tickets per intent (from the Bitext training split) are
placed in the state with their intent; no training. Scored like every other system: Bitext test (EN / NLLB فصحى /
NLLB Najdi), the course's human Arabic, ALLaM dialects; latency one ticket at a time.
Usage: python scripts/eval_d1_fewshot.py --k 1 [--device cuda:0]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.bench import OUT, data, extra_sets, nice, sync  # noqa: E402
from scripts.policy_bench import load_d1  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--k", type=int, default=1)
ap.add_argument("--device", default="cuda:0")
args = ap.parse_args()
name = f"d1_fewshot_k{args.k}"
train, _, test, labels = data()
en = train[train.lang == "en"].groupby("intent").sample(args.k, random_state=0).sample(frac=1, random_state=0)
shots = [{"message": t, "intent": i} for t, i in zip(en.text, en.intent)]
m = load_d1(args.device)
q = {"intent": {"type": "choice", "instructions": "What does the customer want to do? Use the labelled examples as reference.",
                "criteria": {l: nice(l).capitalize() for l in labels}}}


def predict(texts, bs=16):
    out = []
    with torch.inference_mode():
        for s in range(0, len(texts), bs):
            out += [r["answers"]["intent"]["choice"] for r in m.system_one_batch(
                [({"labelled_examples": shots, "message": t}, q) for t in texts[s:s + bs]])]
    return out


ex = extra_sets()
ex.assign(pred=predict(ex.text.tolist()))[["id", "lang", "intent", "pred"]].to_parquet(OUT / f"{name}_extra.parquet")
lat_rows = pd.concat([g.head(100) for _, g in test.groupby("lang")])
predict(lat_rows.text.head(3).tolist())
lat = []
for x in lat_rows.text:
    sync(); t = time.perf_counter(); predict([x]); sync(); lat.append(1000 * (time.perf_counter() - t))
sync(); t = time.perf_counter()
pred = predict(test.text.tolist())
sync(); batch_ms = 1000 * (time.perf_counter() - t) / len(test)
test.assign(pred=pred)[["id", "lang", "intent", "pred"]].to_parquet(OUT / f"{name}.parquet")
json.dump({"method": name, "title": f"d1-3B few-shot ({args.k} English example per intent, {len(shots)} in the prompt)",
           "latency_ms": lat, "batch_ms": batch_ms, "train_s": 0.0, "labels": len(shots),
           "gpu_gb": torch.cuda.max_memory_allocated() / 2**30, "device": torch.cuda.get_device_name(torch.device(args.device))},
          open(OUT / f"{name}.json", "w"))
e = pd.read_parquet(OUT / f"{name}_extra.parquet")
for lang, g in pd.concat([test.assign(pred=pred), e]).groupby("lang"):
    print(lang, "macro-F1", round(100 * f1_score(g.intent, g.pred, average="macro"), 1), flush=True)
c = e[e.lang.str.startswith("course_")]
print("course human Arabic (all)", round(100 * f1_score(c.intent, c.pred, average="macro"), 1), flush=True)
print("p50 latency ms", round(sorted(lat)[len(lat) // 2], 1), flush=True)
