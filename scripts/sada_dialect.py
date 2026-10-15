"""Which Saudi dialect is this? Real speech transcripts from SADA (SDAIA's Saudi Audio Dataset for Arabic).

  test   1,800 SADA test transcripts, 600 per dialect (Najdi / Khaliji (Gulf) / Hijazi), at least 5 words
  train  light: 50 per dialect (150) from SADA's training split
d1-3B zero-shot vs d1-3B + a light LoRA (150 examples, 3 epochs) trained on d1's own readout (the option token's
log-probability at the answer slot). Reports macro-F1, accuracy, per-dialect F1 and the confusion matrix.
Saves the LoRA to runs/d1_lora_sada/. Writes results/sada_dialect.csv.
Usage: python scripts/sada_dialect.py [--device cuda:0] [--per 50] [--epochs 3]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, inject_adapter_in_model
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from transformers import AutoModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ticketassist.decide import MODEL  # noqa: E402
from scripts.finetune_d1 import TARGETS  # noqa: E402

SADA = Path.home() / "semantic-search/data/sada"
OPTS = {"Najdi": "Najdi: central Saudi Arabia (Riyadh, Qassim)",
        "Hijazi": "Hijazi: western Saudi Arabia (Jeddah, Makkah, Madinah)",
        "Khaliji": "Gulf (Khaliji): Eastern Province and the Gulf coast"}
QUESTION = "Which Saudi Arabic dialect is this text written in?"


def load(split, per, seed=0):
    d = pd.read_csv(SADA / f"{split}.csv", header=None, names=["path", "dialect", "text"])
    d = d[d.text.str.split().str.len() >= 5].drop_duplicates("text")
    per = min(per, int(d.dialect.value_counts().min()))  # balanced: as many per dialect as the smallest allows
    return d.groupby("dialect").sample(per, random_state=seed).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--per", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--out", default="runs/d1_lora_sada")
    args = ap.parse_args()
    test = load("test", 600)
    train = load("train", args.per)
    print("test", test.dialect.value_counts().to_dict(), "| train", train.dialect.value_counts().to_dict(), flush=True)
    labels = list(OPTS)

    m = AutoModel.from_pretrained(MODEL, trust_remote_code=True, dtype=torch.bfloat16, device_map={"": args.device}).eval()
    eng = m.engine
    pm = sys.modules[type(eng).__module__.rsplit(".", 1)[0] + ".prompt"]
    q = pm.as_question({"type": "choice", "instructions": QUESTION, "criteria": OPTS})
    groups = pm.readout_ids(eng.tokenizer, q)

    def predict(texts, bs=32):
        out = []
        with torch.inference_mode():
            for s in range(0, len(texts), bs):
                out += [r["answers"]["d"]["choice"] for r in m.system_one_batch(
                    [({"text": t}, {"d": {"type": "choice", "instructions": QUESTION, "criteria": OPTS}}) for t in texts[s:s + bs]])]
        return out

    rows = []

    def score(name, pred, train_s=0.0):
        y = test.dialect.tolist()
        r = {"model": name, "macro-F1": 100 * f1_score(y, pred, average="macro"), "accuracy": 100 * accuracy_score(y, pred),
             **{f"F1 {l}": 100 * f1_score(y, pred, labels=[l], average="macro") for l in labels}, "train s": train_s}
        rows.append(r)
        print({k: round(v, 1) if isinstance(v, float) else v for k, v in r.items()}, flush=True)
        print(pd.DataFrame(confusion_matrix(y, pred, labels=labels), index=[f"true {l}" for l in labels],
                           columns=[f"pred {l}" for l in labels]).to_string(), flush=True)

    score("d1-3B zero-shot", predict(test.text.tolist()))

    inject_adapter_in_model(LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=TARGETS), m)
    params = [p for n, p in m.named_parameters() if "lora_" in n]
    for n, p in m.named_parameters():
        p.requires_grad_("lora_" in n)
    opt = torch.optim.AdamW(params, lr=args.lr)
    steps = len(train) * args.epochs // 4
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=max(1, steps), pct_start=0.1)
    lid = {l: i for i, l in enumerate(labels)}
    m.train()
    t0 = time.time()
    rng = np.random.default_rng(0)
    i = 0
    for ep in range(args.epochs):
        for j in rng.permutation(len(train)):
            r = train.iloc[j]
            ids = eng.tokenizer.encode(eng.render({"text": r.text}, q), add_special_tokens=False)
            lg = m(input_ids=torch.tensor([ids], device=args.device), logits_to_keep=1).logits[0, -1].float()
            logz = lg - torch.logsumexp(lg, -1)
            loss = torch.nn.functional.cross_entropy(torch.stack([logz[g].max() for g in groups])[None],
                                                     torch.tensor([lid[r.dialect]], device=args.device))
            (loss / 4).backward()
            i += 1
            if i % 4 == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad()
        print(f"epoch {ep + 1} done ({time.time() - t0:.0f}s), last loss {loss.item():.3f}", flush=True)
    train_s = time.time() - t0
    m.eval()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    torch.save({n: p.detach().cpu() for n, p in m.named_parameters() if "lora_" in n}, out / "lora.pt")
    json.dump({"targets": TARGETS, "r": 16, "alpha": 32, "examples": len(train), "epochs": args.epochs}, open(out / "lora.json", "w"))
    score(f"d1-3B + LoRA ({len(train)} SADA examples, {args.epochs} epochs)", predict(test.text.tolist()), train_s)
    pd.DataFrame(rows).to_csv(ROOT / f"results/sada_dialect_{Path(args.out).name}.csv", index=False)
    print(pd.DataFrame(rows).round(1).to_markdown(index=False))


if __name__ == "__main__":
    main()
