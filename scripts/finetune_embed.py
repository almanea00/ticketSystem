"""Fine-tune EmbeddingGemma (LoRA) for similar-ticket retrieval, including Arabic/Najdi -> English.

Pairs come from the train split only: a ticket (English, or its فصحى/Najdi translation) and a RELATED
English train ticket (same team + >= 2 specific shared tags, the same rule the test uses).
Loss: InfoNCE over the batch's positives plus one hard negative per pair (a ticket the base model ranks
high that is NOT related). The memory side is the same model, so the ticket memory is re-embedded after.
Validation: val tickets (en/ar/najdi) searching the train memory, MRR@10.

Usage: python scripts/finetune_embed.py [--out runs/ticket_lora] [--steps 1500]
"""

import argparse
import collections
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ticketassist.embed import PROMPTS, Embedder, ticket_text  # noqa: E402
from scripts.eval_retrieve import relevance  # noqa: E402

TARGETS = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"


def encode(model, texts, grad):
    feats = model.preprocess(texts, prompt=PROMPTS["similarity"][0])
    feats = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in feats.items()}
    with torch.set_grad_enabled(grad), torch.autocast("cuda", dtype=torch.bfloat16):
        return F.normalize(model(feats)["sentence_embedding"].float(), dim=-1)


@torch.no_grad()
def val_mrr(model, val, memory_text, rel):
    model.eval()
    D = torch.cat([encode(model, memory_text[s:s + 128], False) for s in range(0, len(memory_text), 128)])
    out = {}
    for lang, g in val.groupby("lang"):
        Q = torch.cat([encode(model, g["text"].tolist()[s:s + 128], False) for s in range(0, len(g), 128)])
        top = (Q @ D.T).topk(10, dim=1).indices.cpu().numpy()
        rr = []
        for row, i in zip(top, g.index):
            first = next((k for k, r in enumerate(row) if r in rel[i]), None)
            rr.append(0 if first is None else 1 / (first + 1))
        out[lang] = round(100 * float(np.mean(rr)), 2)
    out["mean"] = round(float(np.mean(list(out.values()))), 2)
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/ticket_lora")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--temp", type=float, default=0.05)
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)

    d = pd.read_parquet(ROOT / "data/tickets.parquet")
    memory = d[(d["split"] == "train") & (d["lang"] == "en")].reset_index(drop=True)
    memory_text = [ticket_text(s, b) for s, b in zip(memory["subject"], memory["body"])]
    row_of = dict(zip(memory["id"], range(len(memory))))

    queries = d[d["split"] == "train"].reset_index(drop=True)          # en + translated ar/najdi
    qrel, _ = relevance(memory, queries)
    for i, tid in enumerate(queries["id"]):                             # a ticket is not its own match
        qrel[i].pop(row_of[tid], None)
    keep = [i for i, r in enumerate(qrel) if r]
    print(f"train queries with a related ticket: {len(keep)} of {len(queries)}", queries.iloc[keep].groupby("lang").size().to_dict(),
          flush=True)

    val = d[d["split"] == "val"].reset_index(drop=True)
    vrel, _ = relevance(memory, val)
    vkeep = [i for i, r in enumerate(vrel) if r]
    val = val.iloc[vkeep].copy()
    val["text"] = [ticket_text(s, b) for s, b in zip(val["subject"], val["body"])]
    vrel = {i: vrel[i] for i in vkeep}

    emb = Embedder("cuda", "similarity")
    model = emb.model
    # hard negatives: top base-model neighbours that are not related
    base_D = emb.documents(memory["subject"], memory["body"])
    q_text = [ticket_text(s, b) for s, b in zip(queries["subject"], queries["body"])]
    base_Q = emb.queries(queries["subject"].iloc[keep], queries["body"].iloc[keep])
    hard = {}
    for j, i in enumerate(keep):
        s = base_Q[j] @ base_D.T
        own = row_of[queries["id"][i]]
        for r in np.argsort(-s)[:30]:
            if r not in qrel[i] and r != own:
                hard[i] = int(r); break
    print("hard negatives:", len(hard), flush=True)

    model[0].model = get_peft_model(model[0].model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
                                                               target_modules=TARGETS))
    model[0].model.print_trainable_parameters()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    model.train()
    base = val_mrr(model, val, memory_text, vrel)
    print(json.dumps({"step": 0, **base}), flush=True)
    best, run = base["mean"], None
    for step in range(1, args.steps + 1):
        batch, seen = [], set()
        while len(batch) < args.bs:
            i = rng.choice(keep)
            pos = rng.choice(list(qrel[i]))
            if pos in seen:
                continue
            seen.add(pos); batch.append((i, pos, hard.get(i)))
        q = encode(model, [q_text[i] for i, _, _ in batch], True)
        docs = [memory_text[p] for _, p, _ in batch] + [memory_text[h] for _, _, h in batch if h is not None]
        D = encode(model, docs, True)
        doc_rows = [p for _, p, _ in batch] + [h for _, _, h in batch if h is not None]
        s = q @ D.T / args.temp
        # other positives that are also related to a query are not negatives for it
        mask = torch.tensor([[r in qrel[i] and c != k for c, r in enumerate(doc_rows)] for k, (i, _, _) in enumerate(batch)],
                            device="cuda")
        s = s.masked_fill(mask, -1e4)
        loss = F.cross_entropy(s, torch.arange(len(batch), device="cuda"))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        run = loss.item() if run is None else 0.9 * run + 0.1 * loss.item()
        if step % 50 == 0:
            print(json.dumps({"step": step, "loss": round(run, 3)}), flush=True)
        if step % 250 == 0 or step == args.steps:
            v = val_mrr(model, val, memory_text, vrel)
            print(json.dumps({"step": step, **v}), flush=True)
            if v["mean"] > best:
                best = v["mean"]
                model[0].model.save_pretrained(out / "best")
                print("saved best", best, flush=True)
    print(f"best val MRR@10 {best} | base {base['mean']}", flush=True)


if __name__ == "__main__":
    main()
