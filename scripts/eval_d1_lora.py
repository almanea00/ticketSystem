"""Score a saved d1 LoRA (e.g. one trained only on SADA dialect ID) on the intent benchmark, like every other system:
Bitext test (EN / NLLB فصحى / NLLB Najdi), the course's human Arabic, and the ALLaM dialect rewrites.
Writes results/bench/<name>.parquet / _extra.parquet / .json (latency = one ticket at a time, 300 messages).
Usage: python scripts/eval_d1_lora.py --lora runs/d1_lora_sada --name d1_lora_sada150 --title "..." [--device cuda:0]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.bench import OUT, data, extra_sets, nice, sync  # noqa: E402
from scripts.policy_bench import load_d1  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--lora", required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--title", required=True)
ap.add_argument("--device", default="cuda:0")
args = ap.parse_args()
_, _, test, labels = data()
m = load_d1(args.device, args.lora)
q = {"intent": {"type": "choice", "instructions": "What does the customer want to do?",
                "criteria": {l: nice(l).capitalize() for l in labels}}}


def predict(texts, bs=32):
    out = []
    with torch.inference_mode():
        for s in range(0, len(texts), bs):
            out += [r["answers"]["intent"]["choice"] for r in m.system_one_batch([({"message": t}, q) for t in texts[s:s + bs]])]
    return out


ex = extra_sets()
ex.assign(pred=predict(ex.text.tolist()))[["id", "lang", "intent", "pred"]].to_parquet(OUT / f"{args.name}_extra.parquet")
lat_rows = pd.concat([g.head(100) for _, g in test.groupby("lang")])
predict(lat_rows.text.head(5).tolist())
lat = []
for x in lat_rows.text:
    sync(); t = time.perf_counter(); predict([x]); sync(); lat.append(1000 * (time.perf_counter() - t))
sync(); t = time.perf_counter()
pred = predict(test.text.tolist())
sync(); batch_ms = 1000 * (time.perf_counter() - t) / len(test)
test.assign(pred=pred)[["id", "lang", "intent", "pred"]].to_parquet(OUT / f"{args.name}.parquet")
cfg = json.load(open(ROOT / args.lora / "lora.json"))
json.dump({"method": args.name, "title": args.title, "latency_ms": lat, "batch_ms": batch_ms, "train_s": 0.0,
           "labels": 0, "gpu_gb": torch.cuda.max_memory_allocated() / 2**30, "lora_examples": cfg.get("examples"),
           "device": torch.cuda.get_device_name(torch.device(args.device))}, open(OUT / f"{args.name}.json", "w"))
e = pd.read_parquet(OUT / f"{args.name}_extra.parquet")
for lang, g in pd.concat([test.assign(pred=pred), e]).groupby("lang"):
    print(lang, "macro-F1", round(100 * f1_score(g.intent, g.pred, average="macro"), 1), flush=True)
c = e[e.lang.str.startswith("course_")]
print("course human Arabic (all)", round(100 * f1_score(c.intent, c.pred, average="macro"), 1), flush=True)
