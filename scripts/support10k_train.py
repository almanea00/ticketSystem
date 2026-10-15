"""Train the decision models on the same clean training subset of arabic_support_natural_10000.csv, then test again.

Training data: the 487 training messages with no test near-copy (char 3-5-gram cosine < 0.6), the same subset the
LR baselines in support10k_bench.py use; all three questions (intent 40, department 13, priority 3) -> 1,461 examples.
  d1   LoRA r=16 on the language model, loss = cross-entropy over d1's option tokens at the answer slot, 1 epoch
  faisal  LoRA r=16 on the Qwen3.5 language model of Faisal, loss = cross-entropy over the option letters; for the
          40 intents it trains on the 20-option half that holds the right intent (the test uses the same 2-round split)
Then the same 2,000 test messages and metrics as support10k_bench.py; rows are appended to results/support10k.csv.
Saves the d1 LoRA and its questions to runs/d1_lora_support10k/ (lora.pt, lora.json); the ticket system loads it.
Usage: python scripts/support10k_train.py d1|faisal [--device cuda:0]
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, inject_adapter_in_model

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.support10k_bench import TASKS, Tournament, options, split  # noqa: E402


def examples(strict, labels):
    out = []
    for r in strict.itertuples():
        for task, q in TASKS.items():
            out.append(({"message": r.text}, q, options(task, labels[task]), getattr(r, task), task))
    random.Random(0).shuffle(out)
    return out


def train_loop(step_loss, params, n, lr=2e-4, accum=4):
    opt = torch.optim.AdamW(params, lr=lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(1, n // accum), pct_start=0.1)
    t0, run = time.time(), None
    for i in range(n):
        loss = step_loss(i)
        (loss / accum).backward()
        run = loss.item() if run is None else 0.95 * run + 0.05 * loss.item()
        if (i + 1) % accum == 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad()
        if (i + 1) % 200 == 0:
            print(f"{i + 1}/{n} loss {run:.3f} ({time.time() - t0:.0f}s)", flush=True)
    return time.time() - t0


def d1(args, train_ex):
    from scripts.finetune_d1 import TARGETS
    from scripts.policy_bench import D1
    w = D1(args.device)
    m = w.m
    eng = m.engine
    pm = sys.modules[type(eng).__module__.rsplit(".", 1)[0] + ".prompt"]
    cache = {}

    def groups(q, opts):
        k = (q, tuple(opts))
        if k not in cache:
            qq = pm.as_question({"type": "choice", "instructions": q, "criteria": opts})
            cache[k] = (qq, pm.readout_ids(eng.tokenizer, qq), list(opts))
        return cache[k]

    inject_adapter_in_model(LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=TARGETS), m)
    for n, p in m.named_parameters():
        p.requires_grad_("lora_" in n)
    params = [p for n, p in m.named_parameters() if "lora_" in n]
    m.train()

    def step(i):
        state, q, opts, gold, _ = train_ex[i]
        qq, g, keys = groups(q, opts)
        ids = eng.tokenizer.encode(eng.render(state, qq), add_special_tokens=False)
        lg = m(input_ids=torch.tensor([ids], device=args.device), logits_to_keep=1).logits[0, -1].float()
        logz = lg - torch.logsumexp(lg, -1)
        return torch.nn.functional.cross_entropy(torch.stack([logz[x].max() for x in g])[None], torch.tensor([keys.index(gold)], device=args.device))

    secs = train_loop(step, params, len(train_ex))
    m.eval()
    # save the adapter + the exact questions it was trained on, so the system asks the same questions
    out = ROOT / "runs/d1_lora_support10k"
    out.mkdir(parents=True, exist_ok=True)
    torch.save({n: p.detach().cpu() for n, p in m.named_parameters() if "lora_" in n}, out / "lora.pt")
    qs = {}
    for ex in train_ex:      # the full option sets the training used (all 40 intents / 13 teams), not only the labels seen
        qs.setdefault(ex[4], {"type": "choice", "instructions": ex[1], "criteria": ex[2]})
    json.dump({"targets": TARGETS, "r": 16, "alpha": 32, "examples": len(train_ex), "train_seconds": secs,
               "data": "arabic_support_natural_10000.csv (GPT-6.1 Sol), 487 clean training messages x 3 questions",
               "questions": qs}, open(out / "lora.json", "w"), ensure_ascii=False, indent=1)
    print("saved", out, flush=True)
    return w, secs


def faisal(args, train_ex):
    from scripts.policy_bench import FaisalM
    w = FaisalM(args.device)
    f = w.f
    model = f.model
    inject_adapter_in_model(LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                       target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"), model)
    for n, p in model.named_parameters():
        p.requires_grad_("lora_" in n)
    params = [p for n, p in model.named_parameters() if "lora_" in n]
    print("Faisal LoRA params", sum(p.numel() for p in params) / 1e6, "M", flush=True)
    model.train()

    def step(i):
        state, q, opts, gold, _ = train_ex[i]
        items = list(opts.items())
        if len(items) > 24:  # the half that holds the right answer, as in the 2-round test
            items = [items[j::2] for j in range(2)][[k for k, _ in items].index(gold) % 2]
        keys = [k for k, _ in items]
        enc = f.tok([f.prompt(state, q, items)], return_tensors="pt", add_special_tokens=False).to(args.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = model(**enc).logits[0, -1].float()
        return torch.nn.functional.cross_entropy(lg[f.letter_ids[:len(keys)]][None], torch.tensor([keys.index(gold)], device=args.device))

    secs = train_loop(step, params, len(train_ex))
    model.eval()
    return Tournament(w), secs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["d1", "faisal"])
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    _, strict, test = split()
    labels = {t: sorted(test[t].unique()) for t in TASKS}
    train_ex = examples(strict, labels)
    print(args.which, "| training examples", len(train_ex), flush=True)
    runner, secs = (d1 if args.which == "d1" else faisal)(args, train_ex)
    import pandas as pd
    from sklearn.metrics import f1_score
    from scripts.support10k_bench import GROUP
    res = pd.read_csv(ROOT / "results/support10k.csv").to_dict("records")
    name = f"{'d1-3B' if args.which == 'd1' else 'Faisal'} + LoRA (487 messages x 3 tasks)"
    states = [{"message": x} for x in test.text]
    for task, q in TASKS.items():
        t = time.time()
        with torch.inference_mode():
            p = runner.run(states, q, options(task, labels[task]))
        ms = 1000 * (time.time() - t) / len(test)
        r = {"model": name, "type": f"fine-tuned ({secs / 60:.1f} min)", "task": task, "macro-F1": 100 * f1_score(test[task], p, average="macro"),
             "accuracy": 100 * np.mean(np.array(p) == test[task].to_numpy()), "ms/msg (batched)": ms}
        for g, idx in test.groupby("group").groups.items():
            r[g] = 100 * f1_score(test[task][idx], np.array(p)[idx], average="macro", labels=labels[task], zero_division=0)
        res.append(r)
        print({k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items() if k in ("model", "task", "macro-F1", "accuracy")}, flush=True)
        pd.DataFrame(res).to_csv(ROOT / "results/support10k.csv", index=False)
    json.dump({"train_seconds": secs, "examples": len(train_ex)}, open(ROOT / f"results/support10k_{args.which}_lora.json", "w"))


if __name__ == "__main__":
    main()
