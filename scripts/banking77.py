"""Third dataset, for the sandbox showcase: Banking77 (PolyAI, CC-BY-4.0; copy from mteb/banking77).
13,069 online-banking customer questions, 77 fine-grained intents (card_arrival, pending_top_up, ...).

  prep  showcase set: 2 test questions per intent (154), translated to فصحى and Najdi with NLLB-200
        -> data/banking77_showcase.parquet
  eval  d1-3B zero-shot, one 77-option question: full English test (3,076) + the showcase in all three languages
        -> results/banking77_d1.csv

Usage: python scripts/banking77.py prep | eval [--device cuda:0]
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def nice(label: str) -> str:
    return label.replace("_", " ").capitalize()


def question(labels):
    return {"intent": {"type": "choice", "instructions": "What does the bank customer want?",
                       "criteria": {l: nice(l) for l in labels}}}


def prep():
    from scripts.prep_data import Translator
    test = pd.read_parquet(ROOT / "data/banking77_test.parquet")
    show = test.groupby("label_text").sample(2, random_state=0).reset_index(drop=True)  # keeps every column
    show["id"] = [f"K{i:03d}" for i in range(len(show))]
    t = Translator()
    parts = [show.assign(lang="en")]
    for code, lang in (("arb_Arab", "ar"), ("ars_Arab", "najdi")):
        parts.append(show.assign(lang=lang, text=t.texts(show["text"].tolist(), code)))
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(ROOT / "data/banking77_showcase.parquet")
    print(out.groupby("lang").size().to_dict())
    print(out[out["id"] == "K000"][["lang", "text", "label_text"]].to_string())


def evaluate(device):
    import torch
    from sklearn.metrics import accuracy_score, f1_score
    from transformers import AutoModel
    from ticketassist.decide import MODEL
    test = pd.read_parquet(ROOT / "data/banking77_test.parquet")
    show = pd.read_parquet(ROOT / "data/banking77_showcase.parquet")
    labels = sorted(test["label_text"].unique())
    q = question(labels)
    m = AutoModel.from_pretrained(MODEL, trust_remote_code=True, dtype=torch.bfloat16, device_map={"": device}).eval()

    def run(texts):
        out = []
        with torch.no_grad():
            for s in range(0, len(texts), 16):
                out += [r["answers"]["intent"] for r in m.system_one_batch([({"message": x}, q) for x in texts[s:s + 16]])]
        return out

    rows = []
    t = time.time()
    full = run(test["text"].tolist())
    ms = 1000 * (time.time() - t) / len(test)
    pred = [a["choice"] for a in full]
    top5 = [test["label_text"].iloc[i] in sorted(a["probabilities"], key=a["probabilities"].get)[-5:] for i, a in enumerate(full)]
    rows.append({"set": "full test", "lang": "en", "n": len(test), "macro-F1": 100 * f1_score(test["label_text"], pred, average="macro"),
                 "accuracy": 100 * accuracy_score(test["label_text"], pred), "top-5 accuracy": 100 * sum(top5) / len(top5), "ms/message": ms})
    for lang, g in show.groupby("lang"):
        a = run(g["text"].tolist())
        p = [x["choice"] for x in a]
        t5 = [g["label_text"].iloc[i] in sorted(x["probabilities"], key=x["probabilities"].get)[-5:] for i, x in enumerate(a)]
        rows.append({"set": "showcase (2 per intent)", "lang": lang, "n": len(g), "macro-F1": None,
                     "accuracy": 100 * accuracy_score(g["label_text"], p), "top-5 accuracy": 100 * sum(t5) / len(t5), "ms/message": None})
    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "results/banking77_d1.csv", index=False)
    print(res.to_markdown(index=False, floatfmt=".1f"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prep", "eval"])
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    prep() if a.step == "prep" else evaluate(a.device)
