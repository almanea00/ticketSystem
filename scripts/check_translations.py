"""How good are the NLLB-200 translations (English -> فصحى / Najdi) of the test sets?

  meaning kept   EmbeddingGemma cosine between the English message and its translation
                 (cross-lingual; English vs an unrelated translation gives the floor)
  Najdi-ness     share of Najdi outputs identical to the فصحى output, and share using common Najdi words
  breakage       length ratio, leftover Latin words, lost {{placeholders}}
Writes results/translation_check.csv (summary) and results/translation_pairs.parquet (per message).
Usage: python scripts/check_translations.py [--device cuda:0]
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ticketassist.embed import Embedder  # noqa: E402

NAJDI = re.compile(r"(?:^|\s)(?:وش|ايش|إيش|أبي|ابي|أبغى|ابغى|ابغي|يبي|تبي|نبي|ليش|الحين|وين|زين|مب|مو|كذا|ذا|هذي|اللي|عشان|علشان|يمديك|يمدي|وايد|شلون|كيفك|بعد|حق|لين|دحين|ودي|أبيك)(?=\s|$|[،.؟!])")
LATIN = re.compile(r"[A-Za-z]{3,}")
PLACE = re.compile(r"\{\{[^}]+\}\}")


def pairs():
    t = pd.read_parquet(ROOT / "data/tickets.parquet")
    t = t[t["split"] == "test"].assign(text=lambda x: x["subject"].fillna("") + ". " + x["body"])
    b = pd.read_parquet(ROOT / "data/bitext.parquet")
    b = b[b["split"] == "test"]
    out = []
    for name, d in (("Tobi tickets", t), ("Bitext", b)):
        w = d.pivot_table(index="id", columns="lang", values="text", aggfunc="first").dropna()
        w["dataset"] = name
        out.append(w.reset_index())
    return pd.concat(out, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    p = pairs()
    emb = Embedder(args.device, "similarity")
    E = {l: emb.queries([""] * len(p), p[l].tolist()) for l in ("en", "ar", "najdi")}
    p["sim_ar"] = (E["en"] * E["ar"]).sum(1)
    p["sim_najdi"] = (E["en"] * E["najdi"]).sum(1)
    rng = np.random.default_rng(0)
    p["sim_random"] = (E["en"] * E["ar"][rng.permutation(len(p))]).sum(1)   # floor: unrelated pairs
    p["najdi_same_as_msa"] = p["najdi"].str.strip() == p["ar"].str.strip()
    p["najdi_words"] = p["najdi"].map(lambda s: len(NAJDI.findall(s)) > 0)
    p["msa_najdi_words"] = p["ar"].map(lambda s: len(NAJDI.findall(s)) > 0)
    for l in ("ar", "najdi"):
        p[f"len_ratio_{l}"] = p[l].str.split().map(len) / p["en"].str.split().map(len).clip(lower=1)
        p[f"latin_{l}"] = p[l].map(lambda s: len(LATIN.findall(s)) > 0)
    p["placeholders_en"] = p["en"].map(lambda s: len(PLACE.findall(s)))
    p["placeholders_kept_ar"] = p["ar"].map(lambda s: len(PLACE.findall(s)))
    p.to_parquet(ROOT / "results/translation_pairs.parquet")
    rows = []
    for name, g in p.groupby("dataset"):
        has_ph = g["placeholders_en"] > 0
        rows.append({"dataset": name, "messages": len(g),
                     "meaning kept, فصحى (median cos)": g["sim_ar"].median(), "meaning kept, Najdi (median cos)": g["sim_najdi"].median(),
                     "unrelated pair (median cos)": g["sim_random"].median(),
                     "فصحى below 0.5 (%)": 100 * (g["sim_ar"] < 0.5).mean(), "Najdi below 0.5 (%)": 100 * (g["sim_najdi"] < 0.5).mean(),
                     "Najdi identical to فصحى (%)": 100 * g["najdi_same_as_msa"].mean(),
                     "Najdi uses Najdi words (%)": 100 * g["najdi_words"].mean(), "فصحى uses Najdi words (%)": 100 * g["msa_najdi_words"].mean(),
                     "length ratio فصحى / English": g["len_ratio_ar"].median(),
                     "leftover English words, فصحى (%)": 100 * g["latin_ar"].mean(),
                     "placeholders kept (%)": 100 * (g.loc[has_ph, "placeholders_kept_ar"] > 0).mean() if has_ph.any() else None})
    s = pd.DataFrame(rows).set_index("dataset").T
    s.to_csv(ROOT / "results/translation_check.csv")
    print(s.round(2).to_markdown())


if __name__ == "__main__":
    main()
