"""The whole ticket path: triage (d1) -> similar past tickets (cosine, d1 re-rank) -> draft (ALLaM) -> check (d1).

The ticket memory is the English train split: its vectors are cached in data/memory_<mode>.npy.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .decide import Decider
from .embed import Embedder
from .retrieve import CosineIndex

ROOT = Path(__file__).resolve().parent.parent
MIN_COSINE = 0.55      # below this a past ticket is not shown at all
MIN_SAME = 0.5         # d1 P(same problem) needed to count as a similar case
GROUNDED_OK = 0.7      # d1 P(grounded) needed to pass the draft


@dataclass
class Result:
    triage: dict
    cases: list[dict] = field(default_factory=list)       # top similar past tickets, best first
    draft: str = ""
    grounded: float | None = None
    timings: dict = field(default_factory=dict)


class Assistant:
    def __init__(self, device="cuda", draft_device=None, mode="similarity", adapter=None, k=20, top=5):
        t = time.time()
        self.memory = pd.read_parquet(ROOT / "data/tickets.parquet")
        self.memory = self.memory[(self.memory["split"] == "train") & (self.memory["lang"] == "en")].reset_index(drop=True)
        self.embedder = Embedder(device, mode, adapter=adapter)
        cache = ROOT / f"data/memory_{mode}{'_' + Path(adapter).parent.name if adapter else ''}.npy"
        if cache.exists():
            vecs = np.load(cache)
        else:
            vecs = self.embedder.documents(self.memory["subject"], self.memory["body"])
            np.save(cache, vecs)
        self.index = CosineIndex(vecs)
        self.decider = Decider(device)
        self.draft_device = draft_device
        self._drafter = None
        self.k, self.top = k, top
        print(f"assistant ready in {time.time() - t:.0f}s ({len(self.memory)} past tickets)", flush=True)

    @property
    def drafter(self):
        if self._drafter is None and self.draft_device:
            from .draft import Drafter
            self._drafter = Drafter(self.draft_device)
        return self._drafter

    def handle(self, subject: str, body: str, team_filter: bool = False, write_draft: bool = True) -> Result:
        tm = {}
        t = time.time()
        tri = self.decider.triage([(subject, body)])[0]
        tm["triage"] = time.time() - t

        t = time.time()
        q = self.embedder.queries([subject], [body])
        mask = (self.memory["queue"].to_numpy() == tri["queue"])[None, :] if team_filter else None
        cands = self.index.search(q, k=self.k, mask=mask, min_score=MIN_COSINE)[0]
        tm["retrieve"] = time.time() - t

        t = time.time()
        cases = []
        if cands:
            rows = [r for r, _ in cands]
            p = self.decider.same_problem((subject, body), [tuple(self.memory.loc[r, ["subject", "body", "answer"]])
                                                            for r in rows])
            for (r, cos), ps in sorted(zip(cands, p), key=lambda x: -x[1])[:self.top]:
                if ps < MIN_SAME:
                    break
                m = self.memory.loc[r]
                cases.append({"id": m["id"], "queue": m["queue"], "priority": m["priority"], "subject": m["subject"],
                              "body": m["body"], "answer": m["answer"], "cosine": round(cos, 3), "same": round(ps, 3)})
        tm["rerank"] = time.time() - t

        res = Result(tri, cases, timings=tm)
        if write_draft and self.drafter is not None:
            t = time.time()
            res.draft = self.drafter.draft(subject, body, cases)
            tm["draft"] = time.time() - t
            t = time.time()
            res.grounded = self.decider.grounded((subject, body), [{k: c[k] for k in ("id", "subject", "answer")}
                                                                   for c in cases], res.draft)
            tm["check"] = time.time() - t
        return res
