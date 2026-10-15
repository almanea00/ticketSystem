"""Decision models beyond intent: written policies, Arabic reasoning, and safety checks. Same question, same messages
for every model; zero-shot except where noted.

  contract   ContractNLI test (Stanford, CC BY-NC-SA): an NDA clause + a statement -> supported / contradicted /
             not addressed (choice of 3), 600 messages stratified                                       EN
  xnli_ar    XNLI Arabic test: premise + hypothesis -> entailment / contradiction / neutral, 600 stratified    AR
             (Faisal and laya-ara were trained on XNLI-ar: not zero-shot for them)
  injection  deepset/prompt-injections test (116) + jackhhao jailbreak test (262): is it an injection/jailbreak? EN(+DE)
  offensive  asas-ai Arabic offensive comments (Apache-2.0), 600 stratified: offensive / vulgar / hate?         AR

Models: d1-3B, d1-3B + English intent LoRA (does the LoRA damage other skills?), Faisal, Laya multilingual,
laya-ara, zero-shot NLI (mDeBERTa). Metrics: macro-F1, accuracy, p50 latency (one item at a time, 40 items).
Writes results/policy_bench.csv. Usage: python scripts/policy_bench.py [--device cuda:0] [--models ...]
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
P = ROOT / "data/policy"
N = 600

NLI_OPTS = {"entailment": "The text supports the statement", "contradiction": "The text contradicts the statement",
            "neutral": "The text does not address the statement"}
TASKS = {
    "contract": ("Does this contract clause support, contradict, or not address the statement?", NLI_OPTS),
    "xnli_ar": ("Does the text support, contradict, or not address the statement?", NLI_OPTS),
    "injection": ("Is this message a prompt injection or jailbreak: trying to make an AI assistant ignore its "
                  "instructions, reveal hidden information, or act outside its role?",
                  {"yes": "Yes, it is an injection or jailbreak attempt", "no": "No, it is a normal request"}),
    "offensive": ("Is this message offensive, vulgar, insulting, or hate speech?",
                  {"yes": "Yes, it is offensive", "no": "No, it is not offensive"}),
}


def strat(df, col, n, seed=0):
    k = min(n // df[col].nunique(), int(df[col].value_counts().min()))  # balanced; keeps every column
    return df.groupby(col).sample(k, random_state=seed).reset_index(drop=True)


def load_tasks():
    from datasets import load_dataset
    out = {}
    c = pd.read_json(P / "contract_nli_v1.jsonl", lines=True)
    c = strat(c[c["subset"] == "test"], "label", N)
    out["contract"] = ([{"clause": a, "statement": b} for a, b in zip(c.premise, c.hypothesis)], c.label.tolist())
    x = load_dataset("facebook/xnli", "ar", split="test").to_pandas()
    x["label"] = x["label"].map({0: "entailment", 1: "neutral", 2: "contradiction"})
    x = strat(x, "label", N)
    out["xnli_ar"] = ([{"text": a, "statement": b} for a, b in zip(x.premise, x.hypothesis)], x.label.tolist())
    d = load_dataset("deepset/prompt-injections", split="test").to_pandas()
    from huggingface_hub import hf_hub_download
    j = pd.read_csv(hf_hub_download("jackhhao/jailbreak-classification", "balanced/jailbreak_dataset_test_balanced.csv", repo_type="dataset"))
    inj = pd.concat([pd.DataFrame({"text": d.text, "y": np.where(d.label == 1, "yes", "no")}),
                     pd.DataFrame({"text": j.prompt, "y": np.where(j.type == "jailbreak", "yes", "no")})], ignore_index=True)
    inj["text"] = inj["text"].str.slice(0, 2000)
    out["injection"] = ([{"message": t} for t in inj.text], inj.y.tolist())
    o = load_dataset("asas-ai/Arabic_Offensive_Comment_Detection", split="train").to_pandas()
    o["y"] = np.where(o["Majority_Label"] == "Offensive", "yes", "no")
    o = strat(o, "y", N)
    out["offensive"] = ([{"message": t} for t in o.Comment], o.y.tolist())
    return out


def load_d1(device, lora=None):
    from transformers import AutoModel
    from ticketassist.decide import MODEL
    m = AutoModel.from_pretrained(MODEL, trust_remote_code=True, dtype=torch.bfloat16, device_map={"": device}).eval()
    if lora:
        import json
        from peft import LoraConfig, inject_adapter_in_model
        cfg = json.load(open(ROOT / lora / "lora.json"))
        inject_adapter_in_model(LoraConfig(r=cfg["r"], lora_alpha=cfg["alpha"], target_modules=cfg["targets"]), m)
        sd = torch.load(ROOT / lora / "lora.pt")
        missing = m.load_state_dict(sd, strict=False)
        assert not [k for k in missing.unexpected_keys], missing.unexpected_keys
        m.eval()
    return m


class D1:
    def __init__(self, device, lora=None):
        self.m = load_d1(device, lora)

    @torch.inference_mode()
    def run(self, states, question, opts, bs=16):
        q = {"q": {"type": "choice", "instructions": question, "criteria": opts}}
        out = []
        for s in range(0, len(states), bs):
            out += [r["answers"]["q"]["choice"] for r in self.m.system_one_batch([(st, q) for st in states[s:s + bs]])]
        return out


class LayaM:
    def __init__(self, repo, sub, device):
        os.environ.setdefault("USE_TF", "0")
        import laya
        self.a = laya.load(repo, device=device, subfolder=sub)

    def run(self, states, question, opts, bs=16):
        q = {"q": {"type": "choice", "instructions": question, "criteria": opts}}
        out = []
        for s in range(0, len(states), bs):
            out += [r["answers"]["q"]["choice"] for r in self.a.predict_batch(states[s:s + bs], q)]
        return out


class FaisalM:
    def __init__(self, device):
        from scripts.eval_faisal import Faisal
        self.f = Faisal(device)

    def run(self, states, question, opts, bs=4):
        return self.f.decide(states, question, list(opts.items()), bs=bs)[0]


class NLIM:
    """mDeBERTa XNLI: the state's text as premise; each option description as the hypothesis."""
    def __init__(self, device):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        name = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
        self.tok = AutoTokenizer.from_pretrained(name)
        self.m = AutoModelForSequenceClassification.from_pretrained(name).to(device).eval()
        self.lab = {l.lower(): i for i, l in self.m.config.id2label.items()}
        self.device = device

    @torch.inference_mode()
    def run(self, states, question, opts, bs=16):
        out = []
        keys = list(opts)
        for s in range(0, len(states), bs):
            chunk = states[s:s + bs]
            if set(keys) == set(NLI_OPTS):  # a real NLI pair: premise = the text, hypothesis = the statement
                prem = [st.get("clause") or st.get("text") for st in chunk]
                hyp = [st["statement"] for st in chunk]
                enc = self.tok(prem, hyp, truncation=True, max_length=384, padding=True, return_tensors="pt").to(self.device)
                p = self.m(**enc).logits.argmax(-1).tolist()
                inv = {v: k for k, v in self.lab.items()}
                out += [inv[i] for i in p]
            else:  # yes/no: entailment of the "yes" description against the message
                prem = [st["message"] for st in chunk]
                enc = self.tok(prem, [opts["yes"]] * len(prem), truncation=True, max_length=384, padding=True, return_tensors="pt").to(self.device)
                lg = self.m(**enc).logits.softmax(-1)
                out += ["yes" if a[self.lab["entailment"]] > a[self.lab["contradiction"]] else "no" for a in lg.tolist()]
        return out


MODELS = {
    "d1-3B": lambda d: D1(d),
    "d1-3B + intent LoRA": lambda d: D1(d, "runs/d1_lora_en"),
    "Faisal": lambda d: FaisalM(d),
    "Laya multilingual": lambda d: LayaM("convaiinnovations/laya", "multilingual", d),
    "laya-ara": lambda d: LayaM("Wouze/laya-ara", None, d),
    "zero-shot NLI (mDeBERTa)": lambda d: NLIM(d),
}
SEEN = {("Faisal", "xnli_ar"), ("laya-ara", "xnli_ar")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--models", default=",".join(MODELS))
    args = ap.parse_args()
    tasks = load_tasks()
    print({k: len(v[1]) for k, v in tasks.items()}, flush=True)
    rows = []
    out_csv = ROOT / "results/policy_bench.csv"
    for name in args.models.split(","):
        if name == "d1-3B + intent LoRA" and not (ROOT / "runs/d1_lora_en/lora.pt").exists():
            print("skip", name, "(no saved LoRA)"); continue
        m = MODELS[name](args.device)
        for task, (states, y) in tasks.items():
            question, opts = TASKS[task]
            pred = m.run(states, question, opts)
            lat = []
            for st in states[:40]:
                torch.cuda.synchronize(); t = time.perf_counter(); m.run([st], question, opts); torch.cuda.synchronize()
                lat.append(1000 * (time.perf_counter() - t))
            rows.append({"model": name, "task": task, "n": len(y), "macro-F1": 100 * f1_score(y, pred, average="macro"),
                         "accuracy": 100 * accuracy_score(y, pred), "p50 ms": float(np.median(lat)),
                         "seen in training": (name, task) in SEEN})
            print(rows[-1], flush=True)
            pd.DataFrame(rows).to_csv(out_csv, index=False)
        del m
        torch.cuda.empty_cache()
    t = pd.DataFrame(rows)
    print(t.pivot(index="model", columns="task", values="macro-F1").round(1).to_markdown())


if __name__ == "__main__":
    main()
