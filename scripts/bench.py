"""Lab 3.4-style benchmark on Bitext intent (27 classes): every approach on the same 2,449 test messages
in English, فصحى and Najdi, with saved predictions, latency and resource use.

Per method it saves results/bench/<method>.parquet (id, lang, pred) and results/bench/<method>.json:
  latency_ms  one message at a time (as a live ticket arrives), 100 per language after a warm-up
  batch_ms    time per message when the whole test set is run in batches (throughput)
  train_s     training time (0 for zero-shot), labels = labelled messages used, gpu_gb = peak GPU memory
scripts/bench_table.py turns these into the table (macro-F1 + bootstrap 95% CI, p50/p95 latency, cost per 1k).

Training data for every trained method: the 3,598 English train messages + their فصحى and Najdi translations
(10,794); validation (model selection) = the val split in the same three languages. Test messages never train.

Usage: python scripts/bench.py <method> [--device cuda:0]     methods: see METHODS at the bottom
"""

import argparse
import difflib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results/bench"
LAT_PER_LANG = 100


def nice(label):
    return label.replace("_", " ")


def data():
    d = pd.read_parquet(ROOT / "data/bitext.parquet")
    ar = pd.read_parquet(ROOT / "data/bitext_train_ar.parquet")
    cols = ["id", "lang", "text", "intent"]
    test = d[d["split"] == "test"][cols].reset_index(drop=True)
    train = pd.concat([d[(d["split"] == "train") & (d["lang"] == "en")], ar[ar["split"] == "train"]])[cols].reset_index(drop=True)
    val = pd.concat([d[(d["split"] == "val") & (d["lang"] == "en")], ar[ar["split"] == "val"]])[cols].reset_index(drop=True)
    return train, val, test, sorted(d["intent"].unique())


def extra_sets():
    """Extra test sets that are NOT used for training or model selection:
    the course's 108 hand-written Arabic tickets (فصحى + Gulf), and ALLaM's dialect rewrites if present."""
    parts = []
    c = pd.read_csv(ROOT / "data/course/arabic_tickets.csv")
    parts.append(pd.DataFrame({"id": [f"C{i:03d}" for i in range(len(c))], "lang": "course_" + c["dialect"],
                               "text": c["text"], "intent": c["intent"]}))
    dia = ROOT / "data/bitext_dialects.parquet"
    if dia.exists():
        parts.append(pd.read_parquet(dia)[["id", "lang", "text", "intent"]])
    return pd.concat(parts, ignore_index=True)


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ---------------------------------------------------------------- methods
class Method:
    labels_used = 0

    def fit(self, train, val, labels):
        pass

    def predict(self, texts):  # list[str] -> list[label]
        raise NotImplementedError


class TfidfLR(Method):
    def fit(self, train, val, labels):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(train["text"])
        self.lr = LogisticRegression(max_iter=3000, C=5).fit(self.vec.transform(train["text"]), train["intent"])
        self.labels_used = len(train)

    def predict(self, texts):
        return list(self.lr.predict(self.vec.transform(texts)))


class EmbedLR(Method):
    def __init__(self, name, prefix=None, prompt=None, device="cuda:0"):
        from sentence_transformers import SentenceTransformer
        kw = {"config_kwargs": {"vision_config": None, "audio_config": None}} if "gemma" in name else {}
        self.m = SentenceTransformer(name, device=device, model_kwargs={"torch_dtype": torch.bfloat16}, **kw)
        self.prefix, self.prompt = prefix or "", prompt

    def enc(self, texts, bs=64):
        return self.m.encode([self.prefix + t for t in texts], prompt=self.prompt, batch_size=bs, normalize_embeddings=True)

    def fit(self, train, val, labels):
        from sklearn.linear_model import LogisticRegression
        self.lr = LogisticRegression(max_iter=3000, C=10).fit(self.enc(train["text"].tolist()), train["intent"])
        self.labels_used = len(train)

    def predict(self, texts):
        return list(self.lr.predict(self.enc(list(texts))))


class NLI(Method):
    """Zero-shot NLI: each intent becomes the hypothesis "The customer wants to <intent>."."""
    def __init__(self, device="cuda:0"):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        name = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
        self.tok = AutoTokenizer.from_pretrained(name)
        self.m = AutoModelForSequenceClassification.from_pretrained(name).to(device).eval()
        self.entail = [i for i, l in self.m.config.id2label.items() if l.lower().startswith("entail")][0]
        self.device = device

    def fit(self, train, val, labels):
        self.labels = labels
        self.hyp = [f"The customer wants to {nice(l)}." for l in labels]

    @torch.inference_mode()
    def predict(self, texts, bs=8):
        out = []
        for s in range(0, len(texts), bs):
            chunk = list(texts[s:s + bs])
            prem = [t for t in chunk for _ in self.hyp]
            hyp = self.hyp * len(chunk)
            enc = self.tok(prem, hyp, truncation=True, max_length=128, padding=True, return_tensors="pt").to(self.device)
            logit = self.m(**enc).logits[:, self.entail].view(len(chunk), len(self.hyp))
            out += [self.labels[i] for i in logit.argmax(1).tolist()]
        return out


class FewShotLLM(Method):
    """Qwen2.5-1.5B-Instruct, one English example per intent in the prompt, greedy answer mapped to the closest intent."""
    def __init__(self, device="cuda:0"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        name = "Qwen/Qwen2.5-1.5B-Instruct"
        self.tok = AutoTokenizer.from_pretrained(name)
        self.tok.padding_side = "left"
        self.m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map={"": device}).eval()
        self.device = device

    def fit(self, train, val, labels):
        self.labels = labels
        en = train[train["lang"] == "en"]
        shots = en.groupby("intent").sample(1, random_state=0)
        ex = "\n".join(f"Message: {r.text}\nIntent: {r.intent}" for r in shots.itertuples())
        self.system = ("You classify customer-support messages (English or Arabic) into exactly one intent. "
                       "Answer with the intent name only, copied exactly from this list:\n" + ", ".join(labels) +
                       "\n\nExamples:\n" + ex)
        self.labels_used = len(shots)

    def prompt(self, t):
        msgs = [{"role": "system", "content": self.system}, {"role": "user", "content": f"Message: {t}\nIntent:"}]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    @torch.inference_mode()
    def predict(self, texts, bs=16):
        out = []
        for s in range(0, len(texts), bs):
            enc = self.tok([self.prompt(t) for t in texts[s:s + bs]], return_tensors="pt", padding=True).to(self.device)
            g = self.m.generate(**enc, max_new_tokens=10, do_sample=False, pad_token_id=self.tok.eos_token_id)
            for ans in self.tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True):
                a = ans.strip().split()[0].strip(".,'\"`") if ans.strip() else ""
                out.append(a if a in self.labels else (difflib.get_close_matches(a, self.labels, 1, 0) or [self.labels[0]])[0])
        return out


class D1(Method):
    def __init__(self, device="cuda:0"):
        from transformers import AutoModel
        from ticketassist.decide import MODEL
        self.m = AutoModel.from_pretrained(MODEL, trust_remote_code=True, dtype=torch.bfloat16, device_map={"": device}).eval()

    def fit(self, train, val, labels):
        self.q = {"intent": {"type": "choice", "instructions": "What does the customer want to do?",
                             "criteria": {l: nice(l).capitalize() for l in labels}}}

    @torch.inference_mode()
    def predict(self, texts, bs=32):
        out = []
        for s in range(0, len(texts), bs):
            out += [r["answers"]["intent"]["choice"] for r in self.m.system_one_batch([({"message": t}, self.q) for t in texts[s:s + bs]])]
        return out


class Laya(Method):
    def __init__(self, repo, sub=None, device="cuda:0"):
        os.environ.setdefault("USE_TF", "0")
        import laya
        self.agent = laya.load(repo, device=device, subfolder=sub)

    def fit(self, train, val, labels):
        self.q = {"intent": {"type": "choice", "instructions": "What does the customer want to do?",
                             "criteria": {l: nice(l).capitalize() for l in labels}}}

    def predict(self, texts, bs=32):
        out = []
        for s in range(0, len(texts), bs):
            out += [r["answers"]["intent"]["choice"] for r in self.agent.predict_batch([{"message": t} for t in texts[s:s + bs]], self.q)]
        return out


class Faisal(Method):
    """At most 24 options: two groups (14 + 13 intents), then a final between the two winners."""
    def __init__(self, device="cuda:0"):
        from scripts.eval_faisal import Faisal as F
        self.f = F(device)

    def fit(self, train, val, labels):
        opts = [(l, nice(l).capitalize()) for l in labels]
        self.groups = [opts[:14], opts[14:]]
        self.desc = dict(opts)
        self.question = "What does the customer want to do?"

    def predict(self, texts, bs=8):
        states = [{"message": t} for t in texts]
        wins = [self.f.decide(states, self.question, g, bs=bs)[0] for g in self.groups]
        out = []
        for st, a, b in zip(states, *wins):
            out.append(self.f.decide([st], self.question, [(a, self.desc[a]), (b, self.desc[b])], bs=1)[0][0])
        return out


class Faisal27(Method):
    """One pass over all 27 intents with letters A-Z + AA (beyond the trained A-X), options shuffled per message."""
    def __init__(self, device="cuda:0"):
        from scripts.faisal_27 import Faisal27 as F
        self.f = F(device)

    def fit(self, train, val, labels):
        self.labels = labels

    def predict(self, texts):
        return self.f.single(list(texts), self.labels)[0]


class FineTune(Method):
    """Full fine-tune (encoder) or LoRA (decoder LLM) with a classification head; best epoch by val macro-F1."""
    def __init__(self, name, lora=False, epochs=4, lr=5e-5, bs=32, device="cuda:0"):
        self.name, self.lora, self.epochs, self.lr, self.bs, self.device = name, lora, epochs, lr, bs, device

    def fit(self, train, val, labels):
        from sklearn.metrics import f1_score
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
        self.labels = labels
        lid = {l: i for i, l in enumerate(labels)}
        self.tok = AutoTokenizer.from_pretrained(self.name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        torch.manual_seed(0)
        m = AutoModelForSequenceClassification.from_pretrained(self.name, num_labels=len(labels),
                                                               dtype=torch.bfloat16 if self.lora else torch.float32)
        m.config.pad_token_id = self.tok.pad_token_id
        if self.lora:
            from peft import LoraConfig, get_peft_model
            m = get_peft_model(m, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="SEQ_CLS",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
        self.m = m.to(self.device)
        X, y = train["text"].tolist(), train["intent"].map(lid).to_numpy()
        opt = torch.optim.AdamW([p for p in self.m.parameters() if p.requires_grad], lr=self.lr, weight_decay=0.01)
        steps = (len(X) + self.bs - 1) // self.bs * self.epochs
        sched = get_linear_schedule_with_warmup(opt, steps // 10, steps)
        rng = np.random.default_rng(0)
        best, best_state = -1, None
        for ep in range(self.epochs):
            self.m.train()
            order = rng.permutation(len(X))
            for s in range(0, len(X), self.bs):
                b = order[s:s + self.bs]
                enc = self.tok([X[i] for i in b], truncation=True, max_length=64, padding=True, return_tensors="pt").to(self.device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = torch.nn.functional.cross_entropy(self.m(**enc).logits.float(), torch.as_tensor(y[b], device=self.device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in self.m.parameters() if p.requires_grad], 1.0)
                opt.step(); sched.step(); opt.zero_grad()
            f1 = 100 * f1_score(val["intent"], self.predict(val["text"].tolist()), average="macro")
            print(f"epoch {ep + 1}: val macro-F1 {f1:.1f}", flush=True)
            if f1 > best:
                best, best_state = f1, {k: v.detach().clone() for k, v in self.m.state_dict().items()}
        self.m.load_state_dict(best_state)
        self.labels_used = len(train)

    @torch.inference_mode()
    def predict(self, texts, bs=64):
        self.m.eval()
        out = []
        for s in range(0, len(texts), bs):
            enc = self.tok(list(texts[s:s + bs]), truncation=True, max_length=64, padding=True, return_tensors="pt").to(self.device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out += [self.labels[i] for i in self.m(**enc).logits.argmax(-1).tolist()]
        return out


METHODS = {
    "tfidf_lr": ("TF-IDF + logistic regression", lambda d: TfidfLR()),
    "nli_zeroshot": ("Zero-shot NLI (mDeBERTa-v3 XNLI)", lambda d: NLI(d)),
    "e5_lr": ("e5 embeddings + logistic regression", lambda d: EmbedLR("intfloat/multilingual-e5-base", prefix="query: ", device=d)),
    "llm_fewshot": ("LLM few-shot (Qwen2.5 1.5B)", lambda d: FewShotLLM(d)),
    "distilmbert_ft": ("Fine-tuned DistilmBERT", lambda d: FineTune("distilbert/distilbert-base-multilingual-cased", device=d)),
    "qwen05_lora": ("LoRA-tuned LLM (Qwen2.5 0.5B)", lambda d: FineTune("Qwen/Qwen2.5-0.5B-Instruct", lora=True, epochs=3, lr=2e-4, device=d)),
    "embgemma_lr": ("EmbeddingGemma + logistic regression", lambda d: EmbedLR("google/embeddinggemma-2", prompt="task: classification | query: ", device=d)),
    "marbert_ft": ("Fine-tuned MARBERTv2", lambda d: FineTune("UBC-NLP/MARBERTv2", lr=3e-5, device=d)),
    "d1_zeroshot": ("d1-3B zero-shot", lambda d: D1(d)),
    "laya_ml_zeroshot": ("Laya multilingual zero-shot", lambda d: Laya("convaiinnovations/laya", "multilingual", d)),
    "laya_ara_zeroshot": ("laya-ara zero-shot", lambda d: Laya("Wouze/laya-ara", None, d)),
    "faisal_zeroshot": ("Faisal zero-shot (2-round, 24-option limit)", lambda d: Faisal(d)),
    "faisal27_zeroshot": ("Faisal zero-shot (one pass, 27 letters)", lambda d: Faisal27(d)),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("method", choices=list(METHODS))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--extra-only", action="store_true", help="only (re)score the extra test sets")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    train, val, test, labels = data()
    title, make = METHODS[args.method]
    print(title, "| train", len(train), "val", len(val), "test", len(test), flush=True)
    torch.cuda.reset_peak_memory_stats()
    m = make(args.device)
    t = time.time()
    m.fit(train, val, labels)
    train_s = time.time() - t if m.labels_used and m.labels_used > 27 else 0.0
    ex = extra_sets()
    ex.assign(pred=m.predict(ex["text"].tolist()))[["id", "lang", "intent", "pred"]].to_parquet(OUT / f"{args.method}_extra.parquet")
    from sklearn.metrics import f1_score as _f1
    exp = pd.read_parquet(OUT / f"{args.method}_extra.parquet")
    print("extra:", {l: round(100 * _f1(g["intent"], g["pred"], average="macro"), 1) for l, g in exp.groupby("lang")}, flush=True)
    if args.extra_only:
        return

    # latency: one message at a time, 100 per language, after a warm-up
    lat_rows = pd.concat([g.head(LAT_PER_LANG) for _, g in test.groupby("lang")])
    m.predict(lat_rows["text"].head(5).tolist())
    lat = []
    for x in lat_rows["text"]:
        sync(); t = time.perf_counter()
        m.predict([x])
        sync(); lat.append(1000 * (time.perf_counter() - t))

    # throughput + predictions: the whole test set in batches
    sync(); t = time.perf_counter()
    pred = m.predict(test["text"].tolist())
    sync(); batch_ms = 1000 * (time.perf_counter() - t) / len(test)

    test.assign(pred=pred)[["id", "lang", "intent", "pred"]].to_parquet(OUT / f"{args.method}.parquet")
    info = {"method": args.method, "title": title, "latency_ms": lat, "batch_ms": batch_ms, "train_s": train_s,
            "labels": int(m.labels_used), "gpu_gb": torch.cuda.max_memory_allocated() / 2**30,
            "device": torch.cuda.get_device_name(torch.device(args.device))}
    json.dump(info, open(OUT / f"{args.method}.json", "w"))
    from sklearn.metrics import f1_score
    for lang, g in test.assign(pred=pred).groupby("lang"):
        print(lang, "macro-F1", round(100 * f1_score(g["intent"], g["pred"], average="macro"), 1), flush=True)
    print(f"p95 latency {np.percentile(lat, 95):.1f} ms | batch {batch_ms:.2f} ms/msg | train {train_s:.0f} s | "
          f"gpu {info['gpu_gb']:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
