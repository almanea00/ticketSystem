"""Finding similar past tickets.

  cosine   vectors are length 1, so cos(q, d) = q . d: one matrix multiply scores a ticket against all
  bm25     keyword overlap (catches product names, error codes, order numbers)
  hybrid   reciprocal-rank fusion of the two lists
A queue filter (keep only past tickets in the predicted team) and a score threshold are optional.
"""

import re

import numpy as np
from rank_bm25 import BM25Okapi

TOKEN = re.compile(r"[\w؀-ۿ]+")
STOP = set("the a an and or to of in on for is are was were be been it this that i we you my our your me us with "
           "please can could would will have has had not no do does did at as by from about any there their they "
           "hello hi dear team support thank thanks regards best".split())


def tokens(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOP and len(t) > 1]


class CosineIndex:
    def __init__(self, vectors: np.ndarray):
        self.v = vectors.astype(np.float32)

    def search(self, q: np.ndarray, k: int = 20, mask: np.ndarray | None = None, min_score: float = -1.0):
        """q: (n, 768) -> list of [(row, score)] per query, best first."""
        s = q @ self.v.T
        if mask is not None:
            s = np.where(mask, s, -np.inf)
        top = np.argpartition(-s, min(k, s.shape[1] - 1), axis=1)[:, :k]
        out = []
        for i, rows in enumerate(top):
            rows = rows[np.argsort(-s[i, rows])]
            out.append([(int(r), float(s[i, r])) for r in rows if s[i, r] >= min_score])
        return out


class BM25Index:
    def __init__(self, texts: list[str]):
        self.bm25 = BM25Okapi([tokens(t) or ["_"] for t in texts])

    def search(self, texts: list[str], k: int = 20, mask: np.ndarray | None = None):
        out = []
        for i, t in enumerate(texts):
            s = self.bm25.get_scores(tokens(t) or ["_"])
            if mask is not None:
                s = np.where(mask[i], s, -np.inf)
            rows = np.argsort(-s)[:k]
            out.append([(int(r), float(s[r])) for r in rows])
        return out


def rrf(*lists, k: int = 20, c: int = 60):
    """Reciprocal-rank fusion: sum of 1/(c + rank) over the lists each item appears in."""
    out = []
    for per_query in zip(*lists):
        score = {}
        for ranked in per_query:
            for rank, (row, _) in enumerate(ranked):
                score[row] = score.get(row, 0.0) + 1.0 / (c + rank + 1)
        out.append(sorted(score.items(), key=lambda x: -x[1])[:k])
    return out
