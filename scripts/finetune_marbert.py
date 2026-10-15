"""Fine-tune MARBERTv2 (UBC-NLP; Arabic BERT trained on 1B Arabic tweets incl. Gulf/Najdi dialect) as a classifier.

MARBERT is Arabic-only, so it trains on the Arabic training tickets (فصحى + Najdi translations) and is tested on
the Arabic and Najdi test tickets, the same ones d1, Laya and the baselines got.

  tobi    team (10) and priority (3): train = 3,166 Arabic train tickets, val = 1,000, test = 1,626 x 2
  bitext  category (11) and intent (27): Bitext train messages translated to فصحى + Najdi here (cached)

One model per task, best epoch by val macro-F1. Writes results/marbert_<dataset>.csv.
Usage: python scripts/finetune_marbert.py tobi|bitext [--epochs 4] [--device cuda:0]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
MODEL = "UBC-NLP/MARBERTv2"


def tobi():
    d = pd.read_parquet(ROOT / "data/tickets.parquet")
    d = d[d["lang"].isin(["ar", "najdi"])].copy()
    d["text"] = d["subject"].fillna("") + ". " + d["body"]
    return d, {"queue": "team", "priority": "priority"}, 256


def bitext():
    d = pd.read_parquet(ROOT / "data/bitext.parquet")
    cache = ROOT / "data/bitext_train_ar.parquet"
    if not cache.exists():
        from scripts.prep_data import Translator
        tr = d[(d["split"].isin(["train", "val"])) & (d["lang"] == "en")]
        t = Translator()
        parts = []
        for code, lang in (("arb_Arab", "ar"), ("ars_Arab", "najdi")):
            x = tr.copy()
            x["text"] = t.texts(tr["text"].tolist(), code)
            x["lang"] = lang
            parts.append(x)
        pd.concat(parts).to_parquet(cache)
        del t
        torch.cuda.empty_cache()
    d = pd.concat([d[(d["split"] == "test") & d["lang"].isin(["ar", "najdi"])], pd.read_parquet(cache)], ignore_index=True)
    return d, {"category": "category", "intent": "intent"}, 64


class Batches:
    def __init__(self, tok, texts, labels, max_len):
        self.tok, self.texts, self.labels, self.max_len = tok, list(texts), list(labels), max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i], self.labels[i]

    def collate(self, batch):
        enc = self.tok([t for t, _ in batch], truncation=True, max_length=self.max_len, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor([y for _, y in batch])
        return enc


@torch.no_grad()
def predict(model, tok, texts, max_len, device, bs=64):
    model.eval()
    out = []
    for s in range(0, len(texts), bs):
        enc = tok(list(texts[s:s + bs]), truncation=True, max_length=max_len, padding=True, return_tensors="pt").to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(model(**enc).logits.argmax(-1).cpu())
    model.train()
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["tobi", "bitext"])
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    d, tasks, max_len = {"tobi": tobi, "bitext": bitext}[args.dataset]()
    train, val, test = (d[d["split"] == s].reset_index(drop=True) for s in ("train", "val", "test"))
    print({s: len(x) for s, x in (("train", train), ("val", val), ("test", test))}, flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    rows = []
    for task in tasks:
        labels = sorted(d[task].unique())
        lid = {l: i for i, l in enumerate(labels)}
        torch.manual_seed(0)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=len(labels)).to(args.device)
        ds = Batches(tok, train["text"], train[task].map(lid), max_len)
        dl = DataLoader(ds, batch_size=args.bs, shuffle=True, collate_fn=ds.collate)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        sched = get_linear_schedule_with_warmup(opt, len(dl) * args.epochs // 10, len(dl) * args.epochs)
        best, best_state = -1, None
        model.train()
        for ep in range(args.epochs):
            for b in dl:
                b = b.to(args.device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(**b).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
            vp = predict(model, tok, val["text"], max_len, args.device)
            f1 = 100 * f1_score(val[task].map(lid), vp, average="macro")
            print(f"{task} epoch {ep + 1}: loss {loss.item():.3f} val macro-F1 {f1:.1f}", flush=True)
            if f1 > best:
                best, best_state = f1, {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        tp = np.array(labels)[predict(model, tok, test["text"], max_len, args.device)]
        for lang in ("ar", "najdi"):
            m = (test["lang"] == lang).to_numpy()
            rows.append({"method": "MARBERTv2 fine-tuned", "task": task, "lang": lang, "n": int(m.sum()),
                         "macro-F1": 100 * f1_score(test.loc[m, task], tp[m], average="macro", labels=labels, zero_division=0),
                         "accuracy": 100 * accuracy_score(test.loc[m, task], tp[m]), "val macro-F1": best})
        print(pd.DataFrame(rows[-2:]).to_markdown(index=False, floatfmt=".1f"), flush=True)
        del model, opt
        torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(ROOT / f"results/marbert_{args.dataset}.csv", index=False)
    print(pd.DataFrame(rows).to_markdown(index=False, floatfmt=".1f"))


if __name__ == "__main__":
    main()
