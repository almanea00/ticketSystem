"""Tokenizer fertility of d1-3B (and the other models in the system) for Arabic and English.

  fertility        tokens per word (words = whitespace-separated, punctuation stripped); lower is better
  continued words  share of words split into 2+ tokens
  parity           tokens for the Arabic FLORES sentence / tokens for its English translation (1.0 = equal cost)
Text sets: FLORES-200 devtest (the same 1,012 sentences in English and فصحى), the course's human Arabic tickets
(فصحى, Gulf), SADA real speech transcripts (Najdi, Gulf, Hijazi; 500 each), Bitext English tickets (1,000).
Writes results/fertility.csv. Usage: python scripts/fertility.py
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
SS = Path.home() / "semantic-search/data"
TOKENIZERS = {"d1-3B (LFM2.5)": "LiquidAI/d1-3B", "EmbeddingGemma-2": "google/embeddinggemma-2",
              "ALLaM-7B": "ALLaM-AI/ALLaM-7B-Instruct-preview", "Qwen2.5": "Qwen/Qwen2.5-0.5B-Instruct",
              "Faisal (Qwen3.5)": "Omartificial-Intelligence-Space/ara-jev-Faisal",
              "MARBERTv2": "UBC-NLP/MARBERTv2", "e5 / XLM-R": "intfloat/multilingual-e5-base"}
WORD = re.compile(r"[^\W_]+", re.UNICODE)


def text_col(df):
    return df["text"].astype(str)


def sets():
    en = text_col(pd.read_parquet(SS / "flores_plus/eng_Latn.parquet"))
    ar = text_col(pd.read_parquet(SS / "flores_plus/arb_Arab.parquet"))
    c = pd.read_csv(ROOT / "data/course/arabic_tickets.csv")
    s = pd.read_csv(SS / "sada/test.csv", header=None, names=["p", "d", "text"])
    b = pd.read_parquet(ROOT / "data/bitext.parquet")
    b = b[(b.split == "test") & (b.lang == "en")].text.head(1000)
    out = {"FLORES English": en.tolist(), "FLORES فصحى": ar.tolist(),
           "Course human فصحى": c[c.dialect == "msa"].text.tolist(), "Course human Gulf": c[c.dialect == "gulf"].text.tolist(),
           "Bitext English tickets": b.tolist()}
    for d, name in (("Najdi", "SADA Najdi (speech)"), ("Khaliji", "SADA Gulf (speech)"), ("Hijazi", "SADA Hijazi (speech)")):
        out[name] = s[s.d == d].text.head(500).tolist()
    return out, en.tolist(), ar.tolist()


def main():
    data, fl_en, fl_ar = sets()
    rows = []
    for name, repo in TOKENIZERS.items():
        tok = AutoTokenizer.from_pretrained(repo, trust_remote_code="d1" in repo)
        row = {"tokenizer": name, "vocab": len(tok)}
        for set_name, texts in data.items():
            words = [w for t in texts for w in WORD.findall(t)]
            ntok = [len(tok.encode(w, add_special_tokens=False)) for w in words]
            row[f"{set_name}"] = round(float(np.mean(ntok)), 2)
            row[f"{set_name} split %"] = round(100 * float(np.mean(np.array(ntok) > 1)), 1)
        te = [len(tok.encode(t, add_special_tokens=False)) for t in fl_en]
        ta = [len(tok.encode(t, add_special_tokens=False)) for t in fl_ar]
        row["parity AR/EN (FLORES)"] = round(float(np.sum(ta) / np.sum(te)), 2)
        rows.append(row)
        print(name, {k: v for k, v in row.items() if "split" not in k}, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results/fertility.csv", index=False)
    main_cols = ["tokenizer", "vocab"] + list(data) + ["parity AR/EN (FLORES)"]
    print(df[main_cols].to_markdown(index=False))
    print(df[["tokenizer"] + [f"{k} split %" for k in data]].to_markdown(index=False))


if __name__ == "__main__":
    main()
