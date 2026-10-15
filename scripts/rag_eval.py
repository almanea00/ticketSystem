"""Retrieval over the course help center (data/course/kb, DemoMart), Lab 2.3 protocol on data/course/rag_eval.csv.

Chunks = `## ` sections, id "doc#section-slug" (the course's slugify), indexed as "title > section\\ntext".
30 answerable questions (23 EN, 7 AR) + 2 unanswerable. Metrics: Hit@1, Hit@3, MRR (0 if not in the top 20),
Hit@3 per language, and for each reranker how well its top score separates answerable from unanswerable questions.

Systems: BM25 · dense e5-small (course) · dense EmbeddingGemma (ours) · hybrid RRF (BM25 + each dense)
         · + cross-encoder rerank (course: mmarco-mMiniLMv2) · + d1-3B rerank (P(section answers the question))
Also p50 / p95 latency per question (one question at a time, warm-up first) and cost per 1k questions at an assumed
GPU rental price. Writes results/rag_retrieval.csv and results/rag_runs.parquet (per question rankings for the generation step).
Usage: python scripts/rag_eval.py [--device cuda:0]
"""

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data/course"


def slugify(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def load_kb():
    out = []
    for path in sorted((DATA / "kb").glob("*.md")):
        lines = path.read_text(encoding="utf-8").splitlines()
        title, current, buf = lines[0].lstrip("# ").strip(), None, []
        for line in lines[1:] + ["## __end__"]:
            if line.startswith("## "):
                if current and "\n".join(buf).strip():
                    out.append({"id": f"{path.stem}#{slugify(current)}", "doc": path.stem, "title": title,
                                "section": current, "text": "\n".join(buf).strip()})
                current, buf = line[3:].strip(), []
            elif current:
                buf.append(line)
    kb = pd.DataFrame(out)
    kb["index_text"] = kb["title"] + " > " + kb["section"] + "\n" + kb["text"]
    return kb


def metrics(ranks):
    r = np.array([x if x else 0 for x in ranks])
    return {"hit@1": float(np.mean(r == 1)), "hit@3": float(np.mean((r >= 1) & (r <= 3))),
            "mrr": float(np.mean(np.where(r > 0, 1 / np.maximum(r, 1), 0)))}


def rrf(*rankings, c=60, k=20):
    s = {}
    for ranking in rankings:
        for i, d in enumerate(ranking, 1):
            s[d] = s.get(d, 0) + 1 / (c + i)
    return [d for d, _ in sorted(s.items(), key=lambda x: -x[1])][:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--price", type=float, default=0.30, help="GPU rental $/hour (assumption)")
    args = ap.parse_args()
    kb = load_kb()
    ids = kb["id"].tolist()
    q = pd.read_csv(DATA / "rag_eval.csv").fillna({"gold_section": ""})
    missing = set(q["gold_section"]) - set(ids) - {""}
    assert not missing, f"gold sections not in KB: {missing}"
    print(len(kb), "KB sections |", (q.gold_section != "").sum(), "answerable,", (q.gold_section == "").sum(), "unanswerable", flush=True)
    tok = lambda t: re.findall(r"\w+", t.lower())  # noqa: E731

    from sentence_transformers import CrossEncoder, SentenceTransformer
    from transformers import AutoModel
    from ticketassist.decide import MODEL

    bm = BM25Okapi([tok(t) for t in kb["index_text"]])
    e5 = SentenceTransformer("intfloat/multilingual-e5-small", device=args.device)
    D_e5 = e5.encode(["passage: " + t for t in kb["index_text"]], normalize_embeddings=True)
    g = SentenceTransformer("google/embeddinggemma-2", device=args.device, model_kwargs={"torch_dtype": torch.bfloat16},
                            config_kwargs={"vision_config": None, "audio_config": None})
    D_g = g.encode([f"title: {a} > {b} | text: {c}" for a, b, c in zip(kb.title, kb.section, kb.text)], normalize_embeddings=True)
    ce = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", device=args.device)
    d1 = AutoModel.from_pretrained(MODEL, trust_remote_code=True, dtype=torch.bfloat16, device_map={"": args.device}).eval()
    text_of = dict(zip(ids, kb["index_text"]))
    ANS = {"type": "noul", "instructions": "Does this help-center section contain the information needed to answer the customer's question?"}

    # every system: question -> (ranked ids, top score or None); timed one question at a time
    def bm25(x):
        return [ids[i] for i in np.argsort(-bm.get_scores(tok(x)), kind="stable")[:20]], None

    def dense_e5(x):
        v = e5.encode(["query: " + x], normalize_embeddings=True)[0]
        return [ids[i] for i in np.argsort(-(D_e5 @ v))[:20]], None

    def dense_g(x):
        v = g.encode([x], prompt="task: search result | query: ", normalize_embeddings=True)[0]
        return [ids[i] for i in np.argsort(-(D_g @ v))[:20]], None

    def hybrid(dense):
        return lambda x: (rrf(bm25(x)[0], dense(x)[0]), None)

    def ce_rerank(base):
        def f(x):
            cands = base(x)[0]
            s = ce.predict([(x, text_of[c]) for c in cands])
            o = np.argsort(-s)
            return [cands[i] for i in o], float(s[o[0]])
        return f

    @torch.no_grad()
    def d1_rerank(base):
        def f(x):
            cands = base(x)[0]
            r = d1.system_one_batch([({"question": x, "section": text_of[c]}, {"answers": ANS}) for c in cands])
            p = np.array([a["answers"]["answers"]["noul"] for a in r])
            o = np.argsort(-p)
            return [cands[i] for i in o], float(p[o[0]])
        return f

    systems = {
        "BM25": bm25,
        "dense e5-small": dense_e5,
        "dense EmbeddingGemma": dense_g,
        "hybrid RRF (BM25 + e5-small)": hybrid(dense_e5),
        "hybrid RRF (BM25 + EmbeddingGemma)": hybrid(dense_g),
        "hybrid RRF (BM25 + e5-small) + cross-encoder rerank": ce_rerank(hybrid(dense_e5)),
        "hybrid RRF (BM25 + EmbeddingGemma) + cross-encoder rerank": ce_rerank(hybrid(dense_g)),
        "dense EmbeddingGemma + d1 rerank": d1_rerank(dense_g),
        "hybrid RRF (BM25 + EmbeddingGemma) + d1 rerank": d1_rerank(hybrid(dense_g)),
    }
    rank, top_score, lat = {}, {}, {}
    for name, fn in systems.items():
        for x in q.question[:3]:                       # warm-up
            fn(x)
        rank[name], top_score[name], lat[name] = [], [], []
        for x in q.question:
            torch.cuda.synchronize(); t = time.perf_counter()
            r, s = fn(x)
            torch.cuda.synchronize(); lat[name].append(1000 * (time.perf_counter() - t))
            rank[name].append(r); top_score[name].append(s)
        print(name, f"p50 {np.percentile(lat[name], 50):.1f} ms", flush=True)

    ans = q.gold_section != ""
    rows = []
    for name, rk in rank.items():
        rr = [r.index(gs) + 1 if gs in r else None for r, gs in zip(rk, q.gold_section)]
        row = {"system": name, **metrics([x for x, a in zip(rr, ans) if a])}
        for lang in ("en", "ar"):
            m = ans & (q.lang == lang)
            row[f"hit@3 {lang}"] = metrics([x for x, k in zip(rr, m) if k])["hit@3"]
        row["p50 ms"], row["p95 ms"] = float(np.percentile(lat[name], 50)), float(np.percentile(lat[name], 95))
        row["$ / 1k queries"] = float(np.mean(lat[name])) * 1000 / 3.6e6 * args.price
        if top_score[name][0] is not None:
            ts = np.array(top_score[name])
            row["top score answerable (min)"] = float(ts[ans].min())
            row["top score unanswerable (max)"] = float(ts[~ans].max())
            row["separates unanswerable"] = bool(ts[~ans].max() < ts[ans].min())
        rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "results/rag_retrieval.csv", index=False)
    pd.DataFrame({"question": q.question, "lang": q.lang, "gold": q.gold_section,
                  **{k: [" | ".join(r[:5]) for r in v] for k, v in rank.items()}}).to_parquet(ROOT / "results/rag_runs.parquet")
    print(res.round(3).to_markdown(index=False))


if __name__ == "__main__":
    main()
