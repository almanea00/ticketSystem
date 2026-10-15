"""Fast LoRA on d1-3B for the 27-intent question, trained on exactly what d1 decides from: the log-probability of
each option's code token at the answer slot (d1's own prompt + readout), cross-entropy against the right option.

  --data en      50 English Bitext train messages per intent (1,350)
  --data en_ar   the same + their فصحى and Najdi NLLB translations (4,050)
1 epoch, LoRA r=16 on the language model's attention, conv and feed-forward projections, lr 2e-4,
4 examples per step. Model selection: none (one fixed short run). Then the same benchmark scoring as every other
method: Bitext test (EN / NLLB فصحى / NLLB Najdi), course human Arabic, ALLaM dialects, single-ticket latency.

Saves the LoRA weights to runs/d1_lora_<data>/lora.pt (load with load_d1_lora in scripts/policy_bench.py).
Usage: python scripts/finetune_d1.py --data en|en_ar [--device cuda:0]
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
from sklearn.metrics import f1_score
from transformers import AutoModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.bench import OUT, data, extra_sets, nice, sync  # noqa: E402
from ticketassist.decide import MODEL  # noqa: E402

TARGETS = r".*language_model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|out_proj)|feed_forward\.(w1|w2|w3)|conv\.(in_proj|out_proj))"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["en", "en_ar"], default="en")
    ap.add_argument("--per_intent", type=int, default=50)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    name = f"d1_lora_{args.data}"
    train, val, test, labels = data()
    en = train[train["lang"] == "en"].groupby("intent").sample(args.per_intent, random_state=0)
    tr = en if args.data == "en" else pd.concat([en, train[(train["lang"] != "en") & train["id"].isin(set(en["id"]))]])
    tr = tr.sample(frac=1, random_state=0).reset_index(drop=True)
    print(name, "| training examples", len(tr), tr.groupby("lang").size().to_dict(), flush=True)

    model = AutoModel.from_pretrained(MODEL, trust_remote_code=True, dtype=torch.bfloat16, device_map={"": args.device})
    eng = model.engine
    prompt_mod = sys.modules[type(eng).__module__.rsplit(".", 1)[0] + ".prompt"]
    q = prompt_mod.as_question({"type": "choice", "instructions": "What does the customer want to do?",
                                "criteria": {l: nice(l).capitalize() for l in labels}})
    groups = prompt_mod.readout_ids(eng.tokenizer, q)            # token ids per option, max-pooled by d1
    lid = {l: i for i, l in enumerate(labels)}

    inject_adapter_in_model(LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=TARGETS), model)
    params = [p for n, p in model.named_parameters() if "lora_" in n]
    for n, p in model.named_parameters():
        p.requires_grad_("lora_" in n)
    print("trainable LoRA params:", sum(p.numel() for p in params) / 1e6, "M", flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    steps = len(tr) // args.accum
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)

    def option_scores(text):
        ids = eng.tokenizer.encode(eng.render({"message": text}, q), add_special_tokens=False)
        logits = model(input_ids=torch.tensor([ids], device=args.device), logits_to_keep=1).logits[0, -1].float()
        logz = logits - torch.logsumexp(logits, -1)
        return torch.stack([logz[g].max() for g in groups])

    model.train()
    t0, run = time.time(), None
    for i, r in enumerate(tr.itertuples()):
        loss = torch.nn.functional.cross_entropy(option_scores(r.text)[None], torch.tensor([lid[r.intent]], device=args.device))
        (loss / args.accum).backward()
        run = loss.item() if run is None else 0.95 * run + 0.05 * loss.item()
        if (i + 1) % args.accum == 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad()
        if (i + 1) % 200 == 0:
            print(f"{i + 1}/{len(tr)} loss {run:.3f} ({time.time() - t0:.0f}s)", flush=True)
    train_s = time.time() - t0
    model.eval()
    out_dir = ROOT / f"runs/{name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({n: p.detach().cpu() for n, p in model.named_parameters() if "lora_" in n}, out_dir / "lora.pt")
    json.dump({"targets": TARGETS, "r": 16, "alpha": 32, "examples": len(tr)}, open(out_dir / "lora.json", "w"))

    def predict(texts, bs=32):
        out = []
        with torch.inference_mode():
            for s in range(0, len(texts), bs):
                out += [r["answers"]["intent"]["choice"] for r in model.system_one_batch(
                    [({"message": t}, {"intent": {"type": "choice", "instructions": "What does the customer want to do?",
                                                  "criteria": {l: nice(l).capitalize() for l in labels}}}) for t in texts[s:s + bs]])]
        return out

    ex = extra_sets()
    ex.assign(pred=predict(ex["text"].tolist()))[["id", "lang", "intent", "pred"]].to_parquet(OUT / f"{name}_extra.parquet")
    lat_rows = pd.concat([g.head(100) for _, g in test.groupby("lang")])
    predict(lat_rows["text"].head(5).tolist())
    lat = []
    for x in lat_rows["text"]:
        sync(); t = time.perf_counter(); predict([x]); sync(); lat.append(1000 * (time.perf_counter() - t))
    sync(); t = time.perf_counter()
    pred = predict(test["text"].tolist())
    sync(); batch_ms = 1000 * (time.perf_counter() - t) / len(test)
    test.assign(pred=pred)[["id", "lang", "intent", "pred"]].to_parquet(OUT / f"{name}.parquet")
    json.dump({"method": name, "title": f"d1-3B + LoRA ({'English' if args.data == 'en' else 'English + NLLB Arabic'}, {len(tr):,} examples)",
               "latency_ms": lat, "batch_ms": batch_ms, "train_s": train_s, "labels": len(tr),
               "gpu_gb": torch.cuda.max_memory_allocated() / 2**30, "device": torch.cuda.get_device_name(torch.device(args.device))},
              open(OUT / f"{name}.json", "w"))
    e = pd.read_parquet(OUT / f"{name}_extra.parquet")
    for lang, g in pd.concat([test.assign(pred=pred), e]).groupby("lang"):
        print(lang, "macro-F1", round(100 * f1_score(g["intent"], g["pred"], average="macro"), 1), flush=True)
    print(f"train {train_s:.0f}s | p95 {np.percentile(lat, 95):.0f} ms", flush=True)


if __name__ == "__main__":
    main()
