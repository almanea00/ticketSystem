"""Can Faisal pick among all 27 intents in one pass, beyond its trained A-X (24) letters?

On the VALIDATION set (never the test set), 400 messages per language (en / فصحى / Najdi), compare:
  tournament   two groups (14 + 13), then a final between the two winners (3 calls per message)
  single-27    one call, letters A-Z + AA, options shuffled per message as in Faisal's training
For single-27 it also reports accuracy by the letter the right answer sat on: A-X (seen in training) vs Y/Z/AA (unseen),
which shows whether the unseen letters are avoided.
Writes results/faisal_27.csv. Usage: python scripts/faisal_27.py [--device cuda:0] [--n 400]
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.eval_faisal import SYSTEM, Faisal  # noqa: E402
from scripts.bench import data, nice  # noqa: E402

LETTERS27 = [chr(65 + i) for i in range(26)] + ["AA"]
QUESTION = "What does the customer want to do?"


class Faisal27(Faisal):
    def __init__(self, device):
        super().__init__(device)
        self.ids27 = [self.tok.encode(c, add_special_tokens=False)[0] for c in LETTERS27]

    @torch.inference_mode()
    def single(self, texts, labels, seed=0, bs=4):
        rng = random.Random(seed)
        preds, gold_pos = [], []
        for s in range(0, len(texts), bs):
            prompts, orders = [], []
            for t in texts[s:s + bs]:
                order = labels[:]
                rng.shuffle(order)
                orders.append(order)
                task = {"state": {"message": t}, "question": QUESTION,
                        "options": [{"label": LETTERS27[i], "key": k, "description": nice(k).capitalize()} for i, k in enumerate(order)]}
                msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": json.dumps(task, ensure_ascii=False)}]
                prompts.append(self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False))
            enc = self.tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
            p = self.model(**enc).logits[:, -1].float()[:, self.ids27].argmax(1).tolist()
            preds += [o[i] for o, i in zip(orders, p)]
            gold_pos.append(orders)
        return preds, [o for chunk in gold_pos for o in chunk]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n", type=int, default=400)
    args = ap.parse_args()
    _, val, _, labels = data()
    val = pd.concat([g.sample(args.n, random_state=0) for _, g in val.groupby("lang")]).reset_index(drop=True)
    f = Faisal27(args.device)
    opts = [(l, nice(l).capitalize()) for l in labels]
    groups, desc = [opts[:14], opts[14:]], dict(opts)
    st = [{"message": t} for t in val["text"]]

    t = time.time()
    wins = [f.decide(st, QUESTION, g, bs=8)[0] for g in groups]
    tour = [f.decide([s], QUESTION, [(a, desc[a]), (b, desc[b])], bs=1)[0][0] for s, a, b in zip(st, *wins)]
    t_tour = 1000 * (time.time() - t) / len(val)
    t = time.time()
    single, orders = f.single(val["text"].tolist(), labels)
    t_single = 1000 * (time.time() - t) / len(val)

    rows = []
    for name, p, ms in (("tournament (3 calls)", tour, t_tour), ("single pass, 27 letters A-Z+AA", single, t_single)):
        for lang, g in val.assign(pred=p).groupby("lang"):
            rows.append({"variant": name, "lang": lang, "macro-F1": 100 * f1_score(g["intent"], g["pred"], average="macro"),
                         "accuracy": 100 * accuracy_score(g["intent"], g["pred"]), "ms/message": ms})
    pos = np.array([o.index(y) for o, y in zip(orders, val["intent"])])
    ok = np.array(single) == val["intent"].to_numpy()
    picked = np.array([o.index(p) for o, p in zip(orders, single)])
    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "results/faisal_27.csv", index=False)
    print(res.round(1).to_markdown(index=False))
    print(f"\nsingle-27, right answer on a trained letter (A-X): accuracy {100 * ok[pos < 24].mean():.1f}% (n={int((pos < 24).sum())})")
    print(f"single-27, right answer on an unseen letter (Y/Z/AA): accuracy {100 * ok[pos >= 24].mean():.1f}% (n={int((pos >= 24).sum())})")
    print(f"how often Faisal answered Y/Z/AA: {100 * (picked >= 24).mean():.1f}% (fair share would be {100 * 3 / 27:.1f}%)")


if __name__ == "__main__":
    main()
