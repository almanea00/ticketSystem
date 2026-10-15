"""Build the Lab 3.4 table from the saved benchmark predictions (results/bench/*.parquet + *.json).

  Macro-F1 per language with a 95% bootstrap CI (1,000 resamples of the 2,449 test messages)
  p50 / p95 latency  one message at a time on the RTX 3090 (300 messages)
  ms/msg batched     throughput when the whole test set runs in batches
  cost / 1k tickets  batched GPU time x GPU_PRICE (an assumed RTX 3090 cloud rental price, $/hour)
  labels, train time, peak GPU memory

Writes results/bench_table.csv and results/bench_table.md.
Usage: python scripts/bench_table.py [--price 0.30]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
B = ROOT / "results/bench"
ORDER = ["tfidf_lr", "nli_zeroshot", "e5_lr", "llm_fewshot", "distilmbert_ft", "qwen05_lora",
         "embgemma_lr", "marbert_ft", "d1_zeroshot", "laya_ml_zeroshot", "laya_ara_zeroshot", "faisal_zeroshot", "faisal27_zeroshot", "d1_lora_en", "d1_lora_en_ar"]


def boot_f1(y, p, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    labels = np.unique(y)
    point = f1_score(y, p, average="macro", labels=labels)
    s = [f1_score(y[i], p[i], average="macro", labels=labels) for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    lo, hi = np.percentile(s, [2.5, 97.5])
    return 100 * point, 100 * lo, 100 * hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--price", type=float, default=0.30, help="GPU rental, $ per hour (assumption)")
    args = ap.parse_args()
    rows = []
    for m in ORDER:
        if not (B / f"{m}.json").exists():
            continue
        info = json.load(open(B / f"{m}.json"))
        pr = pd.read_parquet(B / f"{m}.parquet")
        row = {"approach": info["title"]}
        for lang, name in (("en", "EN"), ("ar", "AR (فصحى)"), ("najdi", "Najdi")):
            g = pr[pr["lang"] == lang]
            f, lo, hi = boot_f1(g["intent"], g["pred"])
            row[f"F1 {name}"], row[f"CI {name}"] = round(f, 1), f"{lo:.1f}–{hi:.1f}"
        ex = B / f"{m}_extra.parquet"
        if ex.exists():
            e = pd.read_parquet(ex)
            for key, name in (("course_msa", "AR human فصحى"), ("course_gulf", "AR human Gulf"), ("hijazi", "Hijazi (ALLaM)"),
                              ("gulf", "Gulf (ALLaM)"), ("najdi_allam", "Najdi (ALLaM)")):
                g = e[e["lang"] == key]
                if len(g):
                    f, lo, hi = boot_f1(g["intent"], g["pred"])
                    row[f"F1 {name}"], row[f"CI {name}"] = round(f, 1), f"{lo:.1f}–{hi:.1f}"
            ca = e[e["lang"].str.startswith("course_")]
            en = pr[pr["lang"] == "en"]
            row["F1 AR (course, all)"] = round(boot_f1(ca["intent"], ca["pred"])[0], 1)
            both = pd.concat([en[["intent", "pred"]], ca[["intent", "pred"]]])
            row["F1 all (EN test + course AR)"] = round(boot_f1(both["intent"], both["pred"])[0], 1)
        lat = np.array(info["latency_ms"])
        row["p50 ms"], row["p95 ms"] = round(float(np.percentile(lat, 50)), 1), round(float(np.percentile(lat, 95)), 1)
        row["batched ms/msg"] = round(info["batch_ms"], 2)
        row["$ / 1k (batched)"] = round(info["batch_ms"] * 1000 / 3.6e6 * args.price, 5)
        row["$ / 1k (one at a time)"] = round(float(lat.mean()) * 1000 / 3.6e6 * args.price, 4)
        row["labels"] = info["labels"]
        row["train min"] = round(info["train_s"] / 60, 1)
        row["GPU GB"] = round(info["gpu_gb"], 1)
        rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(ROOT / "results/bench_table.csv", index=False)
    show = t[["approach", "F1 EN", "F1 AR (فصحى)", "F1 Najdi", "p95 ms", "$ / 1k (one at a time)", "labels", "train min", "GPU GB"]]
    md = show.to_markdown(index=False)
    dial = [c for c in ["F1 AR human فصحى", "F1 AR human Gulf", "F1 Hijazi (ALLaM)", "F1 Gulf (ALLaM)", "F1 Najdi (ALLaM)"] if c in t]
    if dial:
        md += "\n\nArabic beyond machine translation (course human tickets: 54 فصحى + 54 Gulf; ALLaM rewrites of the Bitext test):\n\n"
        md += t[["approach"] + dial].to_markdown(index=False)
    if "F1 AR (course, all)" in t:
        tpl = pd.DataFrame({"System": t["approach"], "Macro-F1 (all)": t["F1 all (EN test + course AR)"],
                            "Macro-F1 (AR)": t["F1 AR (course, all)"], "p95 latency": t["p95 ms"].astype(str) + " ms",
                            "Cost / 1k": "$" + t["$ / 1k (one at a time)"].astype(str),
                            "Notes": t["labels"].map(lambda n: "zero-shot" if n == 0 else ("few-shot" if n <= 27 else f"trained on {n:,} labels"))})
        md += "\n\nCapstone README format (all = Bitext EN test + course Arabic; AR = course human Arabic, 108 tickets):\n\n" + tpl.to_markdown(index=False)
    cis = t[["approach", "CI EN", "CI AR (فصحى)", "CI Najdi", "p50 ms", "batched ms/msg"]].to_markdown(index=False)
    note = (f"\nCost assumes an RTX 3090 rented at ${args.price:.2f}/hour; latency is one message at a time on that GPU. "
            "CIs are 95% bootstrap intervals over the 2,449 test messages per language; differences inside overlapping "
            "intervals (about 1-2 points) may be noise.\n")
    (ROOT / "results/bench_table.md").write_text(md + "\n\n" + cis + "\n" + note)
    print(md); print(); print(cis); print(note)


if __name__ == "__main__":
    main()
