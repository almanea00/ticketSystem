"""How fast can d1-3B answer one ticket? Same 300 test messages (100 per language) and the same intent question as
the benchmark, one message per call, on the RTX 3090:

  eager            what the benchmark used (system_one_batch, compile off)
  compile          d1's own SystemOne(compile=True): torch.compile with CUDA graphs on the single-question path
  compile + short  the same, options written as the bare intent names (fewer tokens per call)
  batch 8 / 32     several tickets per call (throughput when tickets arrive together)
Each variant's predictions are compared with eager's, so a speed-up cannot silently change the answers.
Writes results/d1_speed.csv. Usage: python scripts/d1_speed.py [--device cuda:0]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.bench import data, nice  # noqa: E402
from ticketassist.decide import MODEL  # noqa: E402


def sync():
    torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    _, _, test, labels = data()
    msgs = pd.concat([g.head(100) for _, g in test.groupby("lang")])
    texts, gold = msgs["text"].tolist(), msgs["intent"].to_numpy()
    q_long = {"intent": {"type": "choice", "instructions": "What does the customer want to do?",
                         "criteria": {l: nice(l).capitalize() for l in labels}}}
    q_short = {"intent": {"type": "choice", "instructions": "Customer intent?", "criteria": {l: l for l in labels}}}

    model = AutoModel.from_pretrained(MODEL, trust_remote_code=True, dtype=torch.bfloat16, device_map={"": args.device}).eval()
    tok = AutoTokenizer.from_pretrained(MODEL)
    SystemOne = sys.modules[type(model).__module__.rsplit(".", 1)[0] + ".runner"].SystemOne
    engines = {"eager": SystemOne(model=model, tokenizer=tok),
               "compile": SystemOne(model=model, tokenizer=tok, compile=True)}

    def single(engine, q):
        lat, pred = [], []
        for t in texts[:8]:                      # warm-up (compile builds its graphs here)
            engine.system_one({"message": t}, q)
        for t in texts:
            sync(); s = time.perf_counter()
            r = engine.system_one({"message": t}, q)
            sync(); lat.append(1000 * (time.perf_counter() - s))
            pred.append(r["answers"]["intent"]["choice"])
        return np.array(lat), np.array(pred), r["usage"]["input_tokens"]

    rows, base = [], None
    for name, eng, q in (("eager", "eager", q_long), ("compile", "compile", q_long), ("compile + short options", "compile", q_short)):
        with torch.inference_mode():
            lat, pred, ntok = single(engines[eng], q)
        base = pred if base is None else base
        rows.append({"variant": name, "p50 ms": np.percentile(lat, 50), "p95 ms": np.percentile(lat, 95),
                     "input tokens": ntok, "accuracy %": 100 * (pred == gold).mean(), "same answer as eager %": 100 * (pred == base).mean()})
        print(rows[-1], flush=True)
    for bs in (8, 32):
        reqs = [({"message": t}, q_long) for t in texts]
        with torch.inference_mode():
            engines["eager"].system_one_batch(reqs[:bs])
            sync(); s = time.perf_counter()
            out = []
            for i in range(0, len(reqs), bs):
                out += engines["eager"].system_one_batch(reqs[i:i + bs])
            sync(); ms = 1000 * (time.perf_counter() - s) / len(reqs)
        pred = np.array([o["answers"]["intent"]["choice"] for o in out])
        rows.append({"variant": f"batch {bs} (per-ticket share)", "p50 ms": ms, "p95 ms": None, "input tokens": None,
                     "accuracy %": 100 * (pred == gold).mean(), "same answer as eager %": 100 * (pred == base).mean()})
        print(rows[-1], flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "results/d1_speed.csv", index=False)
    print(res.round(1).to_markdown(index=False))


if __name__ == "__main__":
    main()
