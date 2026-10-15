"""Tickets -> train / val / test, plus Arabic (فصحى and Najdi) versions made with NLLB-200.

Source: Tobi-Bueck/customer-support-tickets (CC-BY-NC-4.0), English tickets of the 28.6k "5-2-50" file.
Each ticket keeps: subject, body, answer, queue (10 teams), type, priority, tags.
Split 80/10/10, stratified by queue, after dropping duplicate bodies.

Arabic: every test ticket and 4,000 train tickets are translated sentence by sentence
(NLLB is trained on sentences, long paragraphs come out shortened); the test set in both فصحى (arb)
and Najdi (ars), the train subset in a 50/50 mix. Writes data/tickets.parquet with columns
id, split, lang (en | ar | najdi), subject, body, answer, queue, type, priority, tags.

Usage: python scripts/prep_data.py [--no-translate]
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw/aa_dataset-tickets-multi-lang-5-2-50-version.csv"
ARABIC_TRAIN = 4000
SENT = re.compile(r"(?<=[.!?])\s+")


def load():
    d = pd.read_csv(RAW)
    d = d[d["language"] == "en"].copy()
    for c in ("subject", "body", "answer"):
        d[c] = d[c].fillna("").astype(str).str.replace(r"\\n", "\n", regex=True).str.strip()
    d = d[d["body"].str.len() > 20].drop_duplicates("body")
    tags = d[[f"tag_{i}" for i in range(1, 9)]]
    d["tags"] = [sorted({t.strip() for t in row if isinstance(t, str) and t.strip()}) for row in tags.values]
    d = d.reset_index(drop=True)
    d["id"] = [f"T{i:05d}" for i in range(len(d))]
    rng = np.random.default_rng(0)
    d["split"] = "train"
    for _, g in d.groupby("queue"):
        idx = rng.permutation(g.index.to_numpy())
        n = len(idx) // 10
        d.loc[idx[:n], "split"] = "test"
        d.loc[idx[n:2 * n], "split"] = "val"
    d["lang"] = "en"
    return d[["id", "split", "lang", "subject", "body", "answer", "queue", "type", "priority", "tags"]]


class Translator:
    def __init__(self, device="cuda"):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        name = "facebook/nllb-200-distilled-600M"
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(name, src_lang="eng_Latn")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(name, dtype=torch.float16).to(device).eval()
        self.device = device

    def sentences(self, texts, code, bs=64):
        out = []
        for s in range(0, len(texts), bs):
            b = self.tok(texts[s:s + bs], return_tensors="pt", padding=True, truncation=True, max_length=200).to(self.device)
            with self.torch.no_grad():
                g = self.model.generate(**b, forced_bos_token_id=self.tok.convert_tokens_to_ids(code), num_beams=3,
                                        max_new_tokens=160)
            out += self.tok.batch_decode(g, skip_special_tokens=True)
        return out

    def texts(self, texts, code):
        """Paragraphs -> translated paragraphs, translating unique sentences once."""
        pieces = [[s for s in SENT.split(line) if s.strip()] for t in texts for line in [t.replace("\n", " ")]]
        uniq = sorted({s for p in pieces for s in p})
        m = dict(zip(uniq, self.sentences(uniq, code)))
        return [" ".join(m[s] for s in p) for p in pieces]


def main():
    d = load()
    print(d.groupby(["split"]).size().to_dict(), d["queue"].value_counts().to_dict(), flush=True)
    if "--no-translate" in sys.argv:
        d.to_parquet(ROOT / "data/tickets.parquet"); return
    tr = Translator()
    test = d[d["split"] == "test"]
    sub = d[d["split"] == "train"].sample(ARABIC_TRAIN, random_state=0)
    val = d[d["split"] == "val"].sample(500, random_state=0)
    parts = [d]
    for name, rows, code_of in (("test", test, lambda i: ["arb_Arab", "ars_Arab"]),
                                ("val", val, lambda i: ["arb_Arab", "ars_Arab"]),
                                ("train", sub, lambda i: ["arb_Arab" if i % 2 == 0 else "ars_Arab"])):
        for code in ("arb_Arab", "ars_Arab"):
            keep = rows[[code in code_of(i) for i in range(len(rows))]]
            if keep.empty:
                continue
            t = keep.copy()
            t["subject"] = tr.texts(keep["subject"].tolist(), code)
            t["body"] = tr.texts(keep["body"].tolist(), code)
            t["lang"] = "ar" if code == "arb_Arab" else "najdi"
            parts.append(t)
            print(name, code, len(t), "|", t["subject"].iloc[0], "|", t["body"].iloc[0][:200], flush=True)
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(ROOT / "data/tickets.parquet")
    print(out.groupby(["split", "lang"]).size())


if __name__ == "__main__":
    main()
