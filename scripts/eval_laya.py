"""Laya decision models on the same team / priority test as d1 (same questions, same test tickets).

  laya (en)            convaiinnovations/laya, repo root: ModernBERT-large, 421M, English
  laya typed-decisions convaiinnovations/laya/typed-decisions: the fine-tuned English checkpoint
  laya multilingual    convaiinnovations/laya/multilingual: mmBERT-base, 322M, 100+ languages
  laya-ara             Wouze/laya-ara: the multilingual checkpoint fine-tuned on Arabic (CC-BY-NC)
Loaded with laya.load (the library's own code; laya-ara's custom HF code is not used).

Writes results/classify_laya.csv and prints it next to d1 / the baselines from results/classify.csv.
Usage: python scripts/eval_laya.py [--device cuda:0]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
import laya  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ticketassist.decide import PRIORITIES, QUESTIONS, QUEUES, ticket_state  # noqa: E402

MODELS = {"laya (en)": ("convaiinnovations/laya", None), "laya typed-decisions": ("convaiinnovations/laya", "typed-decisions"),
          "laya multilingual": ("convaiinnovations/laya", "multilingual"), "laya-ara": ("Wouze/laya-ara", None)}
TASK_Q = {"queue": "route", "priority": "priority"}


def level_to_label(ans):
    s = ans.get("score")
    if s is None:  # fall back to the most likely level
        probs = ans["probabilities"]
        s = max(probs, key=probs.get)
    return PRIORITIES[min(2, max(0, round(float(s))))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--models", default=",".join(MODELS), help="comma-separated names from MODELS")
    ap.add_argument("--out", default="results/classify_laya.csv")
    args = ap.parse_args()
    d = pd.read_parquet(ROOT / "data/tickets.parquet")
    test = d[d["split"] == "test"].reset_index(drop=True)
    states = [ticket_state(s, b) for s, b in zip(test["subject"], test["body"])]
    rows = []
    for name in args.models.split(","):
        repo, sub = MODELS[name]
        t = time.time()
        agent = laya.load(repo, device=args.device, subfolder=sub)
        print(name, "loaded in", round(time.time() - t, 1), "s", flush=True)
        sample = agent.predict(states[0], QUESTIONS)
        print(json.dumps(sample, default=str, ensure_ascii=False)[:900], flush=True)
        t = time.time()
        out = []
        for s in range(0, len(states), args.bs):
            out += agent.predict_batch(states[s:s + args.bs], QUESTIONS)
            if s % (args.bs * 20) == 0:
                print(f"  {s}/{len(states)}", flush=True)
        speed = 1000 * (time.time() - t) / len(states)
        pred = {"queue": [o["answers"]["route"]["choice"] for o in out],
                "priority": [level_to_label(o["answers"]["priority"]) for o in out]}
        for task, labels in (("queue", list(QUEUES)), ("priority", PRIORITIES)):
            for lang in ("en", "ar", "najdi"):
                m = (test["lang"] == lang).to_numpy()
                y, yp = test.loc[m, task], np.asarray(pred[task])[m]
                rows.append({"method": name, "task": task, "lang": lang, "n": int(m.sum()),
                             "macro-F1": 100 * f1_score(y, yp, average="macro", labels=labels, zero_division=0),
                             "accuracy": 100 * accuracy_score(y, yp), "ms/ticket": speed})
        print(name, f"{speed:.1f} ms/ticket", flush=True)
        pd.DataFrame(rows).to_csv(ROOT / args.out, index=False)  # saved after each model
        del agent
        import torch
        torch.cuda.empty_cache()
    res = pd.DataFrame(rows)
    res.to_csv(ROOT / args.out, index=False)
    base = pd.read_csv(ROOT / "results/classify.csv")
    allr = pd.concat([base, res], ignore_index=True)
    for task in TASK_Q:
        t = allr[allr["task"] == task]
        print(f"\n{task}: macro-F1 (accuracy)")
        print(t.assign(v=t["macro-F1"].round(1).astype(str) + " (" + t["accuracy"].round(1).astype(str) + ")")
              .pivot(index="method", columns="lang", values="v")[["en", "ar", "najdi"]].to_markdown())


if __name__ == "__main__":
    main()
