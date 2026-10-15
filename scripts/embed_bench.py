"""Which (small) embedding model for Arabic with dialects + English text? Same tasks for every model:

  rag        course help-center retrieval (labs/data/rag_eval.csv + kb): Hit@1, Hit@3, MRR; Hit@3 for EN and AR
  intent     Bitext intent via logistic regression trained on ENGLISH ONLY (cross-lingual transfer):
             macro-F1 on the EN test, the course's human فصحى and Gulf tickets
  dialect    MADAR test: a Doha (Gulf) / Cairo / Beirut / Rabat / Tunis sentence finds its English and its MSA
             translation among 2,000 sentences (Hit@1)
  saudi      ALLaM rewrites: a Hijazi / Gulf / Najdi message finds its English original among ~2,400 (Hit@1)
  speed      one sentence at a time, p50 ms on the RTX 3090 and on the CPU; parameters; vector size
Writes results/embed_bench.csv. Usage: python scripts/embed_bench.py [--device cuda:0] [--models a,b]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.bench import data, extra_sets  # noqa: E402
from scripts.rag_eval import load_kb  # noqa: E402

# name: (repo, query prefix, document prefix, extra kwargs)
MODELS = {
    "e5-small": ("intfloat/multilingual-e5-small", "query: ", "passage: ", {}),
    "MiniLM-L12 multilingual": ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "", "", {}),
    "GATE-AraBert-v1": ("Omartificial-Intelligence-Space/GATE-AraBert-v1", "", "", {}),
    "e5-base": ("intfloat/multilingual-e5-base", "query: ", "passage: ", {}),
    "gte-multilingual-base": ("Alibaba-NLP/gte-multilingual-base", "", "", {"trust_remote_code": True}),
    "LaBSE": ("sentence-transformers/LaBSE", "", "", {}),
    "bge-m3": ("BAAI/bge-m3", "", "", {}),
    "EmbeddingGemma-2": ("google/embeddinggemma-2", "task: search result | query: ", "title: none | text: ",
                         {"config_kwargs": {"vision_config": None, "audio_config": None}}),
}
MADAR = Path.home() / "semantic-search/data/madar/madar6.parquet"


def load(repo, kw, device):
    m = SentenceTransformer(repo, device=device, model_kwargs={"torch_dtype": torch.bfloat16 if device != "cpu" else torch.float32}, **kw)
    m.max_seq_length = min(m.max_seq_length or 512, 512)
    return m


def enc(m, texts, prefix="", bs=64):
    return m.encode([prefix + t for t in texts], batch_size=bs, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)


def hit1(Q, D):
    return float(np.mean((Q @ D.T).argmax(1) == np.arange(len(Q))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--models", default=",".join(MODELS))
    args = ap.parse_args()
    train, _, test, labels = data()
    tr_en = train[train["lang"] == "en"]
    te_en = test[test["lang"] == "en"]
    ex = extra_sets()
    course = ex[ex["lang"].str.startswith("course_")]
    saudi = ex[ex["lang"].isin(["hijazi", "gulf", "najdi_allam"])]
    en_of = dict(zip(te_en["id"], te_en["text"]))
    kb = load_kb()
    rag = pd.read_csv(ROOT / "data/course/rag_eval.csv").fillna({"gold_section": ""})
    rag = rag[rag.gold_section != ""].reset_index(drop=True)
    madar = pd.read_parquet(MADAR)
    madar = madar[madar["split"].str.contains("test")].head(2000).reset_index(drop=True)
    rows = []
    for name in args.models.split(","):
        repo, qp, dp, kw = MODELS[name]
        m = load(repo, kw, args.device)
        row = {"model": name, "params (M)": round(sum(p.numel() for p in m.parameters()) / 1e6),
               "dim": m.get_sentence_embedding_dimension()}
        # rag
        D = enc(m, (kb["title"] + " > " + kb["section"] + "\n" + kb["text"]).tolist(), dp)
        S = enc(m, rag.question.tolist(), qp) @ D.T
        ranks = np.array([list(kb["id"][np.argsort(-s)]).index(g) + 1 for s, g in zip(S, rag.gold_section)])
        row["RAG hit@1"], row["RAG hit@3"] = np.mean(ranks == 1), np.mean(ranks <= 3)
        row["RAG MRR"] = np.mean(1 / ranks)
        for lang in ("en", "ar"):
            row[f"RAG hit@3 {lang}"] = np.mean(ranks[rag.lang == lang] <= 3)
        # intent: LR on English only
        lr = LogisticRegression(max_iter=3000, C=10).fit(enc(m, tr_en["text"].tolist(), qp), tr_en["intent"])
        row["intent EN"] = 100 * f1_score(te_en["intent"], lr.predict(enc(m, te_en["text"].tolist(), qp)), average="macro")
        for d in ("course_msa", "course_gulf"):
            g = course[course["lang"] == d]
            row[f"intent {d.replace('course_', 'human ')}"] = 100 * f1_score(g["intent"], lr.predict(enc(m, g["text"].tolist(), qp)), average="macro")
        # dialect matching (MADAR) and Saudi rewrites -> English original
        E = enc(m, madar["English"].tolist(), qp)
        M = enc(m, madar["MSA"].tolist(), qp)
        for city in ("Doha", "Cairo", "Beirut", "Rabat", "Tunis"):
            C = enc(m, madar[city].tolist(), qp)
            row[f"{city}→EN"], row[f"{city}→MSA"] = 100 * hit1(C, E), 100 * hit1(C, M)
        for d in ("hijazi", "gulf", "najdi_allam"):
            g = saudi[saudi["lang"] == d]
            row[f"{d.replace('_allam', '')}→EN"] = 100 * hit1(enc(m, g["text"].tolist(), qp), enc(m, [en_of[i] for i in g["id"]], qp))
        # speed, one sentence at a time
        sample = course["text"].tolist()[:20] + te_en["text"].tolist()[:20]
        for dev in (args.device, "cpu"):
            mm = m if dev == args.device else load(repo, kw, "cpu")
            enc(mm, sample[:3], qp, bs=1)
            t = []
            for x in sample:
                if dev != "cpu":
                    torch.cuda.synchronize()
                s = time.perf_counter(); enc(mm, [x], qp, bs=1)
                if dev != "cpu":
                    torch.cuda.synchronize()
                t.append(1000 * (time.perf_counter() - s))
            row["p50 ms GPU" if dev != "cpu" else "p50 ms CPU"] = float(np.median(t))
            if dev == "cpu":
                del mm
        rows.append(row)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
        del m
        torch.cuda.empty_cache()
        pd.DataFrame(rows).to_csv(ROOT / "results/embed_bench.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_markdown(index=False))


if __name__ == "__main__":
    main()
