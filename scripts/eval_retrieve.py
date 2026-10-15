"""Similar-ticket retrieval: test tickets (English, فصحى, Najdi) search the 13k English train tickets.

A past ticket counts as related when it is in the same team (queue) AND shares at least MIN_SHARED of the
ticket's specific tags (the 6 most common tags, e.g. "IT", "Tech Support", are too generic and ignored).
Graded relevance for nDCG = number of shared specific tags.

Methods:
  bm25                    keywords only
  cosine (search)         EmbeddingGemma, new ticket as query / past tickets as documents
  cosine (similarity)     EmbeddingGemma, "sentence similarity" prompt on both sides
  hybrid                  rank fusion of bm25 + the better cosine
  cosine + team filter    keep only past tickets in the team d1 predicts
  cosine + d1 rerank      d1 "same problem?" over the top 20, on a subset of queries (slower)

Writes results/retrieve.csv. Usage: python scripts/eval_retrieve.py [--rerank 300] [--adapter runs/...]
"""

import argparse
import collections
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ticketassist.embed import Embedder, ticket_text  # noqa: E402
from ticketassist.retrieve import BM25Index, CosineIndex, rrf  # noqa: E402

MIN_SHARED = 2
GENERIC = 6


def relevance(train, test):
    """(test_i -> {train_row: gain}) using queue + specific shared tags."""
    c = collections.Counter(t for ts in train["tags"] for t in ts)
    generic = {t for t, _ in c.most_common(GENERIC)}
    by_queue = collections.defaultdict(list)
    for i, (q, ts) in enumerate(zip(train["queue"], train["tags"])):
        by_queue[q].append((i, set(ts) - generic))
    rel = []
    for q, ts in zip(test["queue"], test["tags"]):
        s = set(ts) - generic
        rel.append({i: len(s & t) for i, t in by_queue[q] if len(s & t) >= MIN_SHARED})
    return rel, generic


def metrics(ranked, rel):
    hit1, hit5, mrr, ndcg = [], [], [], []
    for r, g in zip(ranked, rel):
        rows = [row for row, _ in r]
        first = next((k for k, row in enumerate(rows[:10]) if row in g), None)
        hit1.append(first == 0); hit5.append(first is not None and first < 5)
        mrr.append(1 / (first + 1) if first is not None else 0)
        dcg = sum(g.get(row, 0) / np.log2(k + 2) for k, row in enumerate(rows[:10]))
        ideal = sum(v / np.log2(k + 2) for k, v in enumerate(sorted(g.values(), reverse=True)[:10]))
        ndcg.append(dcg / ideal if ideal else 0)
    return {"Hit@1": 100 * np.mean(hit1), "Hit@5": 100 * np.mean(hit5), "MRR@10": 100 * np.mean(mrr),
            "nDCG@10": 100 * np.mean(ndcg)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerank", type=int, default=300, help="queries per language for the d1 re-rank (0 = skip)")
    ap.add_argument("--adapter", help="LoRA adapter for EmbeddingGemma (fine-tuned retrieval)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    d = pd.read_parquet(ROOT / "data/tickets.parquet")
    train = d[(d["split"] == "train") & (d["lang"] == "en")].reset_index(drop=True)
    test = d[d["split"] == "test"].reset_index(drop=True)
    rel, generic = relevance(train, test)
    has = np.array([len(r) > 0 for r in rel])
    test, rel = test[has].reset_index(drop=True), [r for r, h in zip(rel, has) if h]
    print(f"generic tags ignored: {sorted(generic)} | test tickets with >=1 related past ticket: {len(test)} "
          f"(median {int(np.median([len(r) for r in rel]))} related each)", flush=True)
    langs = {l: (test["lang"] == l).to_numpy() for l in ("en", "ar", "najdi") if (test["lang"] == l).any()}
    q_text = [ticket_text(s, b) for s, b in zip(test["subject"], test["body"])]
    d_text = [ticket_text(s, b) for s, b in zip(train["subject"], train["body"])]

    results = {}
    results["bm25"] = BM25Index(d_text).search(q_text, k=20)
    print("bm25 done", flush=True)
    vecs = {}
    for mode in ("search", "similarity"):
        emb = Embedder(args.device, mode, adapter=args.adapter)
        vecs[mode] = (emb.queries(test["subject"], test["body"]), emb.documents(train["subject"], train["body"]))
        results[f"cosine ({mode})"] = CosineIndex(vecs[mode][1]).search(vecs[mode][0], k=20)
        del emb
        print(mode, "done", flush=True)

    def score(name, ranked, only=None):
        for lang, m in langs.items():
            idx = np.where(m & (only if only is not None else True))[0]
            if len(idx):
                rows.append({"method": name, "lang": lang, "n": len(idx),
                             **metrics([ranked[i] for i in idx], [rel[i] for i in idx])})

    rows = []
    for name, ranked in results.items():
        score(name, ranked)
    best = max(("search", "similarity"),
               key=lambda m: pd.DataFrame([r for r in rows if r["method"] == f"cosine ({m})"])["MRR@10"].mean())
    cos = results[f"cosine ({best})"]
    results["hybrid (bm25 + cosine)"] = rrf(results["bm25"], cos, k=20)
    score("hybrid (bm25 + cosine)", results["hybrid (bm25 + cosine)"])

    import torch
    torch.cuda.empty_cache()
    from ticketassist.decide import Decider
    dec = Decider(args.device)
    tri = dec.triage(list(zip(test["subject"], test["body"])))
    pred_q = np.array([t["queue"] for t in tri])
    mask = pred_q[:, None] == train["queue"].to_numpy()[None, :]
    q, D = vecs[best]
    score(f"cosine + d1 team filter", CosineIndex(D).search(q, k=20, mask=mask))
    print("team filter done; d1 team accuracy on these:", round(100 * np.mean(pred_q == test["queue"].to_numpy()), 1), flush=True)

    if args.rerank:
        rng = np.random.default_rng(0)
        pick = np.zeros(len(test), bool)
        for lang, m in langs.items():
            idx = np.where(m)[0]
            pick[rng.choice(idx, min(args.rerank, len(idx)), replace=False)] = True
        reranked = [None] * len(test)
        for n, i in enumerate(np.where(pick)[0]):
            cands = cos[i]
            p = dec.same_problem((test["subject"][i], test["body"][i]),
                                 [(train["subject"][r], train["body"][r], train["answer"][r]) for r, _ in cands])
            reranked[i] = sorted(zip([r for r, _ in cands], p), key=lambda x: -x[1])
            if n % 100 == 0:
                print("rerank", n, int(pick.sum()), flush=True)
        score(f"cosine ({best}) [same queries]", cos, only=pick)
        score("cosine + d1 rerank", [r if r is not None else [] for r in reranked], only=pick)

    df = pd.DataFrame(rows)
    out = ROOT / ("results/retrieve.csv" if not args.adapter else f"results/retrieve_{Path(args.adapter).parent.name}.csv")
    df.to_csv(out, index=False)
    print(df.pivot_table(index="method", columns="lang", values="MRR@10", sort=False).to_markdown(floatfmt=".1f"))
    print(df.to_markdown(index=False, floatfmt=".1f"))


if __name__ == "__main__":
    main()
