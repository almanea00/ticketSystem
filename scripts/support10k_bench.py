"""Benchmark on arabic_support_natural_10000.csv (user-provided; 10,000 synthetic Arabic support messages, 40 intents,
20 dialects incl. Najdi / Hijazi / Eastern, 13 departments, 3 priorities; marked native_review_required).

Test: 2,000 messages (50 per intent, seed 0). Three decisions: intent (40), department (13), priority (3).
  zero-shot   d1-3B, d1-3B + Bitext intent LoRA, Faisal (2 rounds when > 24 options), Laya multilingual, laya-ara
  trained     TF-IDF + LR, e5-base + LR, EmbeddingGemma-2 + LR on the other 8,000 messages, minus every training message
              within character 3-5-gram cosine 0.6 of a test message (the data is templated: group_id)
Macro-F1 overall and per dialect group. Writes results/support10k.csv and results/support10k_preds.parquet.
Usage: python scripts/support10k_bench.py [--device cuda:0] [--models ...]
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
F = ROOT / "data/support10k/arabic_support_natural_10000.csv"
GROUP = {"sa_najdi": "Saudi Najdi", "sa_hijazi": "Saudi Hijazi", "sa_eastern": "Saudi Eastern", "msa": "فصحى",
         "kw_urban": "Gulf (other)", "qa_urban": "Gulf (other)", "ae_urban": "Gulf (other)", "bh_urban": "Gulf (other)", "om_muscat": "Gulf (other)",
         "lb_beirut": "Levant", "jo_amman": "Levant", "ps_urban": "Levant", "sy_damascus": "Levant",
         "eg_cairo": "Egypt / Sudan", "sd_khartoum": "Egypt / Sudan", "iq_baghdad": "Iraq / Yemen", "ye_sanaa": "Iraq / Yemen",
         "ma_casablanca": "Maghreb", "dz_algiers": "Maghreb", "tn_tunis": "Maghreb"}
TASKS = {"intent": "What does the customer want?", "department": "Which department should handle this message?",
         "priority": "How urgent is this message for the business?"}
PRIO = {"low": "Low: routine issue, no urgent impact", "medium": "Medium: delivery, fulfilment or billing problem that affects the customer",
        "high": "High: security, unauthorized money activity or a physical safety hazard"}


def nice(x):
    return x.replace("_", " ").capitalize()


def options(task, labels):
    return PRIO if task == "priority" else {l: nice(l) for l in labels}


def split():
    d = pd.read_csv(F)
    d["group"] = d.dialect.map(GROUP)
    test = d.groupby("intent").sample(50, random_state=0)
    train = d.drop(test.index)
    from sklearn.feature_extraction.text import TfidfVectorizer
    v = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2).fit(d.text)
    best = np.asarray((v.transform(train.text) @ v.transform(test.text).T).max(1).todense()).ravel()
    strict = train[best < 0.6]
    return train, strict, test.reset_index(drop=True)


class Tournament:
    """Faisal sees at most 24 options: split, pick a winner per group, then a final."""
    def __init__(self, f):
        self.f = f

    def run(self, states, question, opts):
        items = list(opts.items())
        if len(items) <= 24:
            return self.f.run(states, question, opts)
        groups = [dict(items[i::2]) for i in range(2)]
        wins = [self.f.run(states, question, g) for g in groups]
        return [self.f.run([st], question, {a: opts[a], b: opts[b]})[0] for st, a, b in zip(states, *wins)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--models", default="tfidf,e5,embgemma,d1,d1_lora,laya_ml,laya_ara,faisal")
    args = ap.parse_args()
    train, strict, test = split()
    print(f"test {len(test)} | train {len(train)} | strict train (no test near-copies) {len(strict)}", flush=True)
    labels = {t: sorted(test[t].unique()) for t in TASKS}
    preds = pd.DataFrame({"text": test.text, "dialect": test.dialect, "group": test.group, **{t: test[t] for t in TASKS}})
    rows = []

    def record(name, kind, task, p, ms):
        preds[f"{name}|{task}"] = p
        r = {"model": name, "type": kind, "task": task, "macro-F1": 100 * f1_score(test[task], p, average="macro"),
             "accuracy": 100 * np.mean(np.array(p) == test[task].to_numpy()), "ms/msg (batched)": ms}
        for g, idx in test.groupby("group").groups.items():
            r[g] = 100 * f1_score(test[task][idx], np.array(p)[idx], average="macro", labels=labels[task], zero_division=0)
        rows.append(r)
        print({k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items() if k in ("model", "task", "macro-F1", "accuracy")}, flush=True)
        pd.DataFrame(rows).to_csv(ROOT / "results/support10k.csv", index=False)
        preds.to_parquet(ROOT / "results/support10k_preds.parquet")

    want = args.models.split(",")
    from sklearn.linear_model import LogisticRegression
    if "tfidf" in want:
        from sklearn.feature_extraction.text import TfidfVectorizer
        for task in TASKS:
            v = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2).fit(strict.text)
            lr = LogisticRegression(max_iter=3000, C=5, class_weight="balanced").fit(v.transform(strict.text), strict[task])
            t = time.time(); p = lr.predict(v.transform(test.text)); ms = 1000 * (time.time() - t) / len(test)
            record("TF-IDF + LR", f"trained ({len(strict):,})", task, p, ms)
    from sentence_transformers import SentenceTransformer
    for key, name, repo, kw, prompt, pre in (("e5", "e5-base + LR", "intfloat/multilingual-e5-base", {}, None, "query: "),
                                              ("embgemma", "EmbeddingGemma-2 + LR", "google/embeddinggemma-2",
                                               {"config_kwargs": {"vision_config": None, "audio_config": None}}, "task: classification | query: ", "")):
        if key not in want:
            continue
        m = SentenceTransformer(repo, device=args.device, model_kwargs={"torch_dtype": torch.bfloat16}, **kw)
        enc = lambda x: m.encode([pre + s for s in x], prompt=prompt, batch_size=64, normalize_embeddings=True)  # noqa: E731
        Etr = enc(strict.text.tolist())
        t = time.time(); Ete = enc(test.text.tolist()); ms = 1000 * (time.time() - t) / len(test)
        for task in TASKS:
            lr = LogisticRegression(max_iter=3000, C=10, class_weight="balanced").fit(Etr, strict[task])
            record(name, f"trained ({len(strict):,})", task, lr.predict(Ete), ms)
        del m
        torch.cuda.empty_cache()

    from scripts.policy_bench import D1, FaisalM, LayaM
    os.environ.setdefault("USE_TF", "0")
    makers = {"d1": ("d1-3B", lambda: D1(args.device)), "d1_lora": ("d1-3B + Bitext intent LoRA", lambda: D1(args.device, "runs/d1_lora_en")),
              "laya_ml": ("Laya multilingual", lambda: LayaM("convaiinnovations/laya", "multilingual", args.device)),
              "laya_ara": ("laya-ara", lambda: LayaM("Wouze/laya-ara", None, args.device)),
              "faisal": ("Faisal", lambda: Tournament(FaisalM(args.device)))}
    states = [{"message": x} for x in test.text]
    for key in [k for k in makers if k in want]:
        name, make = makers[key]
        mm = make()
        for task, question in TASKS.items():
            t = time.time(); p = mm.run(states, question, options(task, labels[task])); ms = 1000 * (time.time() - t) / len(test)
            record(name, "zero-shot", task, p, ms)
        del mm
        torch.cuda.empty_cache()
    t = pd.DataFrame(rows)
    print(t.pivot_table(index="model", columns="task", values="macro-F1", sort=False).round(1).to_markdown())
    print(t[t.task == "intent"][["model"] + sorted(set(GROUP.values()))].round(1).to_markdown(index=False))


if __name__ == "__main__":
    main()
