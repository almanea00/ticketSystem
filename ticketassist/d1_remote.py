"""d1 served by vLLM: the same `system_one` / `system_one_batch` answers as the local model, over HTTP.

d1 decides by reading the log-probabilities of each option's tokens at one answer slot. Every question is rendered
with d1's own prompt code (prompt.py, shipped with the model) into the exact token ids d1 reads locally, then sent
to vLLM's /v1/completions for ONE token with `allowed_token_ids` = the question's option tokens and their
log-probabilities returned. Restricting the next token to the options adds the same constant to every option's
log-probability, which d1's final softmax cancels, so the answer is d1's. The server must run with
`--logprobs-mode processed_logprobs`, `--max-logprobs` at least the largest option set (400 is plenty) and
`--no-enable-prefix-caching`: with prefix caching on, vLLM 0.28 gives wrong answers for batched d1 requests (its
convolution-layer state is reused across requests); one-by-one requests were unaffected.
Questions on one state are independent rows in d1 too ("mathematically each row alone"); vLLM's prefix cache reads
the shared state once.
"""

import importlib.util
import math
from pathlib import Path

import httpx


def _load_prompt_module(model_dir: Path):
    spec = importlib.util.spec_from_file_location("d1_prompt", model_dir / "prompt.py")
    mod = importlib.util.module_from_spec(spec)
    import sys
    sys.modules["d1_prompt"] = mod   # dataclasses look their module up while the file runs
    spec.loader.exec_module(mod)
    return mod


def _answer(pm, q, probs):
    """d1's api.answer: a noul's P(yes); a choice's pick and its probabilities; a score's expected level."""
    if isinstance(q, pm.Noul):
        return {"type": "noul", "noul": probs[0]}
    best = max(range(len(probs)), key=probs.__getitem__)
    if isinstance(q, pm.Choice):
        names = list(q.criteria)
        return {"type": "choice", "choice": names[best], "confidence": probs[best], "probabilities": dict(zip(names, probs))}
    return {"type": "score", "score": sum(i * p for i, p in enumerate(probs)), "confidence": probs[best],
            "probabilities": {str(i): p for i, p in enumerate(probs)}, "legend": {str(i): t for i, t in enumerate(q.criteria)}}


class RemoteD1:
    """Drop-in for the local d1 model's `system_one` / `system_one_batch`. `model_dir` holds the tokenizer and
    d1's prompt.py (the merged checkpoint folder works)."""

    def __init__(self, base_url: str, model: str, model_dir, api_key: str = ""):
        from transformers import AutoTokenizer
        model_dir = Path(model_dir)
        self.pm = _load_prompt_module(model_dir)
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        bos = getattr(self.tok, "bos_token", None)
        self.bos = bos if isinstance(bos, str) else ""
        self.lead = self.pm.default_lead("lfm2_vl")
        self.model = model
        self.http = httpx.Client(base_url=base_url.rstrip("/"), timeout=120,
                                 headers={"Authorization": f"Bearer {api_key}"} if api_key else {})

    def _rows(self, state, questions):
        out = []
        for name, q in questions.items():
            q = self.pm.as_question(q)
            text = self.pm.render(self.tok, state, q, self.bos, self.lead, self.pm.DEFAULT_STATE_STYLE,
                                  self.pm.DEFAULT_SYSTEM, "desc")
            ids = self.tok.encode(text, add_special_tokens=False)
            groups = self.pm.readout_ids(self.tok, q)
            out.append((name, q, ids, groups))
        return out

    def system_one_batch(self, requests):
        rows = [(i, r) for i, (state, qs, *_) in enumerate(requests) for r in self._rows(state, qs)]
        # one request per question, sent together; each restricted to its own option tokens
        payloads = []
        for _, (name, q, ids, groups) in rows:
            toks = sorted({t for grp in groups for t in grp})
            payloads.append({"model": self.model, "prompt": ids, "max_tokens": 1, "temperature": 0,
                             "logprobs": len(toks), "allowed_token_ids": toks, "return_tokens_as_token_ids": True})
        outs = [None] * len(payloads)
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(16, len(payloads) or 1)) as ex:
            for k, resp in enumerate(ex.map(lambda p: self.http.post("/completions", json=p), payloads)):
                resp.raise_for_status()
                outs[k] = resp.json()["choices"][0]["logprobs"]["top_logprobs"][0]
        answers = [dict() for _ in requests]
        for (i, (name, q, ids, groups)), top in zip(rows, outs):
            lp = {int(k.split(":", 1)[1]): v for k, v in top.items()}
            scores = [max(lp.get(t, -math.inf) for t in grp) for grp in groups]
            m = max(scores)
            ex_ = [math.exp(s - m) for s in scores]
            answers[i][name] = _answer(self.pm, q, [e / sum(ex_) for e in ex_])
        return [{"answers": a, "usage": {"input_tokens": 0, "output_tokens": 0}} for a in answers]

    def system_one(self, state, questions, images=None):
        return self.system_one_batch([(state, questions)])[0]


class SystemOneAPI:
    """A hosted decision API that speaks the same System One format as d1: POST {base}/systemone with
    {"model", "state", "questions"} and get {"answers": {name: {"type", "noul" | "choice", "probabilities", ...}}}.
    Works with TypeSafe's Jev (https://api.typesafe.ai/v1, model jev-latest) and any server with that format."""

    def __init__(self, base_url: str, model: str, api_key: str = "", path: str = "/systemone"):
        if not api_key and "localhost" not in base_url and "127.0.0.1" not in base_url:
            raise ValueError("TA_DECISION=systemone needs TA_DECISION_API_KEY (or TYPESAFE_API_KEY)")
        self.model, self.path = model, path
        self.http = httpx.Client(base_url=base_url.rstrip("/"), timeout=60,
                                 headers={"Authorization": f"Bearer {api_key}"} if api_key else {})

    def system_one(self, state, questions, images=None):
        body = {"model": self.model, "state": state, "questions": questions}
        r = self.http.post(self.path, json=body)
        if r.status_code >= 400:
            raise RuntimeError(f"decision API error {r.status_code}: {r.text[:300]}")
        out = r.json()
        return {"answers": out["answers"], "usage": out.get("usage", {})}

    def system_one_batch(self, requests):
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(8, len(requests) or 1)) as ex:
            return list(ex.map(lambda r: self.system_one(r[0], r[1]), requests))
