"""One d1 LoRA for every decision: continue from runs/d1_lora_support10k (copy kept in runs/d1_lora_support10k_backup)
on the hand-made guard set (data/guard/guard_small.csv, 160 rows x 3 yes/no guard questions = 480 examples) plus a
replay of the 1,461 support examples (intent, department, priority) so those are not forgotten. 1 epoch, same settings.
The guard questions are exactly the ones the system asks (ticketassist.guard.scope_questions with the row's scope).
Usage: python scripts/train_all_lora.py [--source runs/d1_lora_support10k] [--out runs/d1_lora_all] [--guard all|train]
  --guard all   = the final LoRA (all 160 safeguard messages); train = only the train half (the comparison variant)."""
import argparse, json, random, sys, time
from pathlib import Path
import pandas as pd, torch
from peft import LoraConfig, inject_adapter_in_model
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.finetune_d1 import TARGETS  # noqa: E402
from scripts.policy_bench import D1  # noqa: E402
from scripts.support10k_bench import TASKS, options, split  # noqa: E402
from scripts.support10k_train import examples, train_loop  # noqa: E402
from ticketassist.guard import scope_questions  # noqa: E402



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="runs/d1_lora_support10k")
    ap.add_argument("--out", default="runs/d1_lora_all")
    ap.add_argument("--guard", choices=["all", "train"], default="all")
    args = ap.parse_args()
    SRC, OUT = ROOT / args.source, ROOT / args.out
    _, strict, test = split()
    labels = {t: sorted(test[t].unique()) for t in TASKS}
    ex = [(st, {"type": "choice", "instructions": q, "criteria": opts}, list(opts).index(gold)) for st, q, opts, gold, _ in examples(strict, labels)]
    g = pd.read_csv(ROOT / "data/guard/guard_small.csv")
    if args.guard == "train":
        g = g[g.split == "train"]   # the test half is never trained on
    n_guard = 0
    for r in g.itertuples():
        qs = scope_questions(r.scope)
        yes = {"support": r.label == "in_scope", "injection": r.label == "prompt_injection", "harmful": r.label == "harmful"}
        for k, q in qs.items():
            ex.append(({"message": r.text}, q, 0 if yes[k] else 1)); n_guard += 1   # noul readout groups: [yes, no]
    random.Random(1).shuffle(ex)
    print("examples", len(ex), "guard", n_guard, flush=True)

    w = D1("cuda:0"); m = w.m; eng = m.engine
    pm = sys.modules[type(eng).__module__.rsplit(".", 1)[0] + ".prompt"]
    inject_adapter_in_model(LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=TARGETS), m)
    res = m.load_state_dict(torch.load(SRC / "lora.pt"), strict=False); assert not res.unexpected_keys
    for n, p in m.named_parameters():
        p.requires_grad_("lora_" in n)
    params = [p for n, p in m.named_parameters() if "lora_" in n]
    m.train()
    cache = {}

    def step(i):
        st, q, gold = ex[i]
        key = json.dumps(q, sort_keys=True, ensure_ascii=False)
        if key not in cache:
            qq = pm.as_question(q); cache[key] = (qq, pm.readout_ids(eng.tokenizer, qq))
        qq, grp = cache[key]
        ids = eng.tokenizer.encode(eng.render(st, qq), add_special_tokens=False)
        lg = m(input_ids=torch.tensor([ids], device="cuda:0"), logits_to_keep=1).logits[0, -1].float()
        logz = lg - torch.logsumexp(lg, -1)
        return torch.nn.functional.cross_entropy(torch.stack([logz[x].max() for x in grp])[None], torch.tensor([gold], device="cuda:0"))

    secs = train_loop(step, params, len(ex))
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({n: p.detach().cpu() for n, p in m.named_parameters() if "lora_" in n}, OUT / "lora.pt")
    cfg = json.load(open(SRC / "lora.json"))
    cfg.update(examples=len(ex), train_seconds=secs, started_from=str(SRC.relative_to(ROOT)),
               data=cfg["data"] + f" + data/guard/guard_small.csv (hand-made, {len(g)} rows x 3 guard questions)")
    json.dump(cfg, open(OUT / "lora.json", "w"), ensure_ascii=False, indent=1)
    print("saved", OUT, f"{secs / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
