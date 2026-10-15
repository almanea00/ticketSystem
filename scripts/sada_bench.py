"""SADA Saudi-dialect benchmark: which dialect is this real speech transcript in (Najdi / Gulf (Khaliji) / Hijazi)?
Test: 1,500 SADA test transcripts (500 per dialect, >= 5 words). Training text: SADA's full train split
(data/sada_text/sada_text.parquet, text + dialect columns only).

  no training   d1 zero-shot, d1 few-shot (5 per dialect in the prompt), d1 + intent LoRA (the system's model),
                Faisal, Laya multilingual, laya-ara
  trained       char 2-5-gram TF-IDF + LR on ALL Najdi/Gulf/Hijazi train lines (class-balanced);
                EmbeddingGemma-2 + LR and e5-base + LR on 3,000 lines per dialect
Metrics: macro-F1, accuracy, per-dialect F1, p50 latency (one sentence at a time, 40 sentences).
Writes results/sada_bench.csv. Usage: python scripts/sada_bench.py [--device cuda:0]
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.sada_dialect import OPTS, QUESTION  # noqa: E402

LABELS = list(OPTS)


def data(per_test=500, per_emb=3000):
    d = pd.read_parquet(ROOT / "data/sada_text/sada_text.parquet").rename(columns={"cleaned_text": "text", "speaker_dialect": "dialect"})
    d = d[d.dialect.isin(LABELS) & (d.text.fillna("").str.split().str.len() >= 5)].drop_duplicates("text")
    test = d[d.split == "test"].groupby("dialect").sample(per_test, random_state=0).reset_index(drop=True)
    train = d[(d.split == "train") & ~d.text.isin(set(test.text))]
    return train, test, train.groupby("dialect").sample(per_emb, random_state=0)


def score(rows, name, kind, y, pred, lat, n_train):
    rows.append({"method": name, "type": kind, "train lines": n_train, "macro-F1": 100 * f1_score(y, pred, average="macro"),
                 "accuracy": 100 * accuracy_score(y, pred), **{f"F1 {l}": 100 * f1_score(y, pred, labels=[l], average="macro") for l in LABELS},
                 "p50 ms": float(np.median(lat)) if lat else None})
    print({k: (round(v, 1) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)


def timed(fn, texts):
    lat = []
    for t in texts[:40]:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        s = time.perf_counter(); fn([t])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        lat.append(1000 * (time.perf_counter() - s))
    return lat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    train, test, emb_train = data()
    y, X = test.dialect.tolist(), test.text.tolist()
    print("test", test.dialect.value_counts().to_dict(), "| train", train.dialect.value_counts().to_dict(), flush=True)
    rows = []

    # --- trained baselines
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    v = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=3, max_features=400_000).fit(train.text)
    lr = LogisticRegression(max_iter=2000, C=5, class_weight="balanced").fit(v.transform(train.text), train.dialect)
    f = lambda t: lr.predict(v.transform(t))  # noqa: E731
    score(rows, "char n-gram + LR", "trained", y, f(X), timed(f, X), len(train))
    from sentence_transformers import SentenceTransformer
    for name, repo, kw, prompt in (("EmbeddingGemma-2 + LR", "google/embeddinggemma-2", {"config_kwargs": {"vision_config": None, "audio_config": None}}, "task: classification | query: "),
                                   ("e5-base + LR", "intfloat/multilingual-e5-base", {}, None)):
        m = SentenceTransformer(repo, device=args.device, model_kwargs={"torch_dtype": torch.bfloat16}, **kw)
        pre = "query: " if "e5" in repo else ""
        enc = lambda t: m.encode([pre + x for x in t], prompt=prompt, batch_size=64, normalize_embeddings=True)  # noqa: E731
        clf = LogisticRegression(max_iter=3000, C=10).fit(enc(emb_train.text.tolist()), emb_train.dialect)
        f = lambda t: clf.predict(enc(t))  # noqa: E731
        score(rows, name, "trained", y, f(X), timed(f, X), len(emb_train))
        del m
        torch.cuda.empty_cache()

    # --- decision models, no training
    from scripts.policy_bench import FaisalM, LayaM, load_d1
    qd = {"d": {"type": "choice", "instructions": QUESTION, "criteria": OPTS}}

    def d1_runner(m, shots=None):
        q = qd if not shots else {"d": {"type": "choice", "instructions": QUESTION + " Use the labelled examples as reference.", "criteria": OPTS}}

        @torch.inference_mode()
        def run(texts, bs=16):
            out = []
            for s in range(0, len(texts), bs):
                st = [({"labelled_examples": shots, "text": t} if shots else {"text": t}) for t in texts[s:s + bs]]
                out += [r["answers"]["d"]["choice"] for r in m.system_one_batch([(x, q) for x in st])]
            return out
        return run

    m = load_d1(args.device)
    f = d1_runner(m)
    score(rows, "d1-3B zero-shot", "no training", y, f(X), timed(f, X), 0)
    ex = train.groupby("dialect").sample(5, random_state=1).sample(frac=1, random_state=0)
    f = d1_runner(m, [{"text": t, "dialect": d} for t, d in zip(ex.text, ex.dialect)])
    score(rows, "d1-3B few-shot (5 per dialect)", "no training", y, f(X), timed(f, X), 0)
    del m
    torch.cuda.empty_cache()
    m = load_d1(args.device, "runs/d1_lora_en")
    f = d1_runner(m)
    score(rows, "d1-3B + intent LoRA (system model)", "no training", y, f(X), timed(f, X), 0)
    del m
    torch.cuda.empty_cache()
    os.environ.setdefault("USE_TF", "0")
    for name, make in (("Faisal", lambda: FaisalM(args.device)),
                       ("Laya multilingual", lambda: LayaM("convaiinnovations/laya", "multilingual", args.device)),
                       ("laya-ara", lambda: LayaM("Wouze/laya-ara", None, args.device))):
        mm = make()
        f = lambda t: mm.run([{"text": x} for x in t], QUESTION, OPTS)  # noqa: E731
        score(rows, name, "no training", y, f(X), timed(f, X), 0)
        del mm
        torch.cuda.empty_cache()
    t = pd.DataFrame(rows)
    t.to_csv(ROOT / "results/sada_bench.csv", index=False)
    print(t.round(1).to_markdown(index=False))


if __name__ == "__main__":
    main()
