"""Evaluate a predictions file the way the course's capstone/evaluate.py does: overall and per-slice macro-F1 with
a bootstrap 95% CI, the most frequent confusions, and a per-class report.

    python eval/evaluate.py eval/predictions/<system>.csv [--slice lang] [--slice dialect] [--where testset=course_arabic]
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def bootstrap_ci(y_true, y_pred, metric, n=500, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    s = [metric(y_true[i], y_pred[i]) for i in (rng.integers(0, len(y_true), len(y_true)) for _ in range(n))]
    lo, hi = np.percentile(s, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(metric(y_true, y_pred)), float(lo), float(hi)


def summarize(df):
    f1, lo, hi = bootstrap_ci(df.y_true.astype(str), df.y_pred.astype(str), macro_f1)
    return {"n": len(df), "accuracy": round(accuracy_score(df.y_true, df.y_pred), 3), "macro_f1": round(f1, 3),
            "macro_f1_95ci": f"[{lo:.3f}, {hi:.3f}]"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions")
    ap.add_argument("--slice", action="append", default=[])
    ap.add_argument("--where", action="append", default=[], help="column=value filter (repeatable)")
    args = ap.parse_args()
    df = pd.read_csv(args.predictions)
    for w in args.where:
        k, v = w.split("=", 1)
        df = df[df[k].astype(str) == v]
    df["y_pred"] = df["y_pred"].fillna("(abstain)")
    rows = [{"slice": "all", **summarize(df)}]
    for col in args.slice:
        for value, g in df.groupby(col):
            rows.append({"slice": f"{col}={value}", **summarize(g)})
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nMost frequent confusions:")
    c = df[df.y_true != df.y_pred][["y_true", "y_pred"]].value_counts().head(10).rename("count").reset_index()
    print(c.to_string(index=False))
    print()
    print(classification_report(df.y_true, df.y_pred, digits=3, zero_division=0))


if __name__ == "__main__":
    main()
