"""Evaluate the scope guard (ticketassist/guard.py) in its three modes on a CSV with columns scope,text,label[,trick]
(label: in_scope | out_of_scope | prompt_injection), or on a built-in sanity set when no CSV is given:
  in_scope          400 tickets from arabic_support_natural_10000.csv (10 per intent), store scope
  prompt_injection  deepset/prompt-injections test (label 1) + jackhhao jailbreak test, store scope
  out_of_scope      deepset benign + jackhhao benign prompts (general requests, not store support), store scope
Metrics per mode: 3-way macro-F1, false refusals (in_scope blocked), let through (attacks / off-topic passed),
per-label recall, per-trick catch rate (CSV only), p50 latency. The few-shot examples are never test items.
Usage: python scripts/eval_guard.py [--csv path] [--modes zeroshot_noul,zeroshot_choice,fewshot_choice]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ticketassist.guard import INJECT_NO, INJECT_YES, SHOTS, STORE_SCOPE, SUPPORT_NO, SUPPORT_YES  # noqa: E402
from ticketassist.system import TicketSystem  # noqa: E402

LABELS = ["in_scope", "out_of_scope", "prompt_injection"]


def sanity_set():
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    t = pd.read_csv(ROOT / "data/support10k/arabic_support_natural_10000.csv").groupby("intent").sample(10, random_state=21)
    d = load_dataset("deepset/prompt-injections", split="test").to_pandas()
    j = pd.read_csv(hf_hub_download("jackhhao/jailbreak-classification", "balanced/jailbreak_dataset_test_balanced.csv", repo_type="dataset"))
    rows = [(STORE_SCOPE, x, "in_scope", "real ticket (" + dl + ")") for x, dl in zip(t.text, t.dialect)]
    rows += [(STORE_SCOPE, x, "prompt_injection", "deepset injection") for x in d[d.label == 1].text]
    rows += [(STORE_SCOPE, x[:2000], "prompt_injection", "jailbreak") for x in j[j.type == "jailbreak"].prompt]
    rows += [(STORE_SCOPE, x, "out_of_scope", "deepset general request") for x in d[d.label == 0].text]
    rows += [(STORE_SCOPE, x[:2000], "out_of_scope", "general prompt") for x in j[j.type == "benign"].prompt]
    return pd.DataFrame(rows, columns=["scope", "text", "label", "trick"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--modes", default="zeroshot_noul,zeroshot_choice,fewshot_choice")
    args = ap.parse_args()
    df = pd.read_csv(args.csv) if args.csv else sanity_set()
    shots = {t for _, t, _ in SHOTS} | set(SUPPORT_YES + SUPPORT_NO + INJECT_YES + INJECT_NO)
    df = df[~df.text.isin(shots)].reset_index(drop=True)   # never test on the few-shot examples
    if "trick" not in df:
        df["trick"] = ""
    df["trick"] = df["trick"].fillna("")
    print("items", len(df), df.label.value_counts().to_dict(), flush=True)
    s = TicketSystem(draft=False)
    rows = []
    for mode in args.modes.split(","):
        pred, lat = [], []
        for sc, tx in zip(df.scope, df.text):
            t = time.perf_counter()
            pred.append(s.guard(tx, mode=mode, scope=sc)["label"])
            lat.append(1000 * (time.perf_counter() - t))
        df[mode] = pred
        y, p = df.label.to_numpy(), np.array(pred)
        r = {"mode": mode, "macro-F1": 100 * f1_score(y, p, average="macro", labels=LABELS),
             "false refusals (in_scope blocked) %": 100 * np.mean(p[y == "in_scope"] != "in_scope"),
             "let through (attack/off-topic passed) %": 100 * np.mean(p[y != "in_scope"] == "in_scope"),
             **{f"recall {l}": 100 * np.mean(p[y == l] == l) for l in LABELS if (y == l).any()},
             "p50 ms": float(np.median(lat))}
        rows.append(r)
        print({k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    res = pd.DataFrame(rows)
    out = ROOT / ("results/guard_eval.csv" if not args.csv else f"results/guard_eval_{Path(args.csv).stem}.csv")
    res.to_csv(out, index=False)
    df.to_csv(out.with_name(out.stem + "_items.csv"), index=False)
    print(res.round(1).to_markdown(index=False))
    # which tricks / slices get through or get wrongly blocked, per mode
    blocked_ok = lambda g, m: np.mean((g[m] == "in_scope") == (g.label == "in_scope"))  # noqa: E731
    by = df.groupby("trick").apply(lambda g: pd.Series({"n": len(g), **{m: round(100 * blocked_ok(g, m), 1) for m in args.modes.split(",")}}))
    print("\npass/block decision correct (%), per trick or slice:\n" + by.sort_values("n", ascending=False).head(40).to_markdown())


if __name__ == "__main__":
    main()
