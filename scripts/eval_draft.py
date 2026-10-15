"""Draft replies: does grounding in retrieved past tickets help? ALLaM-7B with the top cases vs with none.

On N test tickets per language (English, فصحى, Najdi):
  grounded      d1 P(every claim supported by the cited past tickets)        (with-cases only)
  cites         share of drafts that cite at least one [ID]                    (with-cases only)
  right lang    reply written in the ticket's language (Arabic script for ar/najdi)
  answer sim    EmbeddingGemma cosine between the draft and the real agent answer of that ticket
                (cross-lingual for Arabic tickets: the real answers are English)
Also writes results/drafts_for_rating.csv: blind pairs (A/B order shuffled) for the team's 1-5 human rating;
which one had cases is in results/drafts_rating_key.csv (don't show it to raters).

Usage: python scripts/eval_draft.py [--n 30] [--device cuda:0]
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ticketassist.draft import Drafter, language_of  # noqa: E402
from ticketassist.pipeline import Assistant  # noqa: E402

CITE = re.compile(r"\[T\d{5}\]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="test tickets per language")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    a = Assistant(args.device, draft_device=None)
    drafter = Drafter(args.device)
    d = pd.read_parquet(ROOT / "data/tickets.parquet")
    test = d[d["split"] == "test"]
    ids = test[test["lang"] == "en"]["id"].sample(args.n, random_state=1)
    test = test[test["id"].isin(set(ids))].reset_index(drop=True)
    gold = dict(zip(d[(d["lang"] == "en")]["id"], d[(d["lang"] == "en")]["answer"]))

    rows = []
    for i, t in test.iterrows():
        res = a.handle(t["subject"], t["body"], write_draft=False)
        for setup, cases in (("with cases", res.cases), ("no cases", [])):
            draft = drafter.draft(t["subject"], t["body"], cases, lang=t["lang"])
            g = a.decider.grounded((t["subject"], t["body"]),
                                   [{k: c[k] for k in ("id", "subject", "answer")} for c in cases], draft) if cases else None
            rows.append({"id": t["id"], "lang": t["lang"], "setup": setup, "n_cases": len(cases), "draft": draft,
                         "grounded": g, "cites": bool(CITE.search(draft)),
                         "right_lang": (language_of(draft) == "ar") == (t["lang"] != "en"),
                         "subject": t["subject"], "body": t["body"], "gold": gold[t["id"]]})
        print(i, t["lang"], len(res.cases), "cases", flush=True)
    df = pd.DataFrame(rows)
    v_draft = a.embedder.queries([""] * len(df), df["draft"].tolist())
    v_gold = a.embedder.queries([""] * len(df), df["gold"].tolist())
    df["answer_sim"] = (v_draft * v_gold).sum(1)
    df.to_parquet(ROOT / "results/drafts.parquet")

    summary = df.groupby(["setup", "lang"]).agg(n=("id", "size"), grounded=("grounded", "mean"), cites=("cites", "mean"),
                                                right_lang=("right_lang", "mean"), answer_sim=("answer_sim", "mean"),
                                                cases=("n_cases", "mean")).reset_index()
    summary.to_csv(ROOT / "results/draft.csv", index=False)
    print(summary.to_markdown(index=False, floatfmt=".2f"))

    rng = np.random.default_rng(0)
    blind = []
    for tid, g in df.groupby(["id", "lang"]):
        w, n = g[g["setup"] == "with cases"].iloc[0], g[g["setup"] == "no cases"].iloc[0]
        first = rng.random() < 0.5
        A, B = (w, n) if first else (n, w)
        blind.append({"id": tid[0], "lang": tid[1], "subject": w["subject"], "ticket": w["body"],
                      "reply_A": A["draft"], "reply_B": B["draft"], "rating_A (1-5)": "", "rating_B (1-5)": "",
                      "_A_is": A["setup"]})
    blind = pd.DataFrame(blind)
    blind.drop(columns="_A_is").to_csv(ROOT / "results/drafts_for_rating.csv", index=False)  # raters see this
    blind[["id", "lang", "_A_is"]].to_csv(ROOT / "results/drafts_rating_key.csv", index=False)  # kept hidden


if __name__ == "__main__":
    main()
