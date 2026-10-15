"""Can one d1 with a LoRA always on do every decision? base d1 (no LoRA) vs the support LoRA (runs/d1_lora_support10k_backup)
vs the all-decisions LoRA (runs/d1_lora_all, also trained on the hand-made guard set), same questions:
  guard   scope-aware 3 questions, block if p_injection or p_harmful >= 0.1 -- Gemini v1 (200/label) + hand-made (160)
  answerable  "do these 3 sections answer the message?" (one call over the top 3) -- rag_eval 30 answerable + 2 unanswerable
              + the 20 uncovered store questions of no_answer_test.py
  intent/team  the 10k test set (400 messages) -- the LoRA's own job, for reference
-> results/lora_everywhere.csv"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, torch
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ticketassist.guard import scope_questions  # noqa: E402
from ticketassist.system import TicketSystem  # noqa: E402
from scripts.no_answer_test import UNCOVERED  # noqa: E402
from scripts.support10k_bench import split  # noqa: E402

s = TicketSystem(draft=False)
ANS3 = {"a": {"type": "noul", "instructions": "Do these help-center sections contain the information needed to answer the customer's message?"}}
g1 = pd.read_csv(ROOT / "data/guard/guardrail_decision_dataset_5000.csv").groupby("label").sample(200, random_state=1)
g2 = pd.read_csv(ROOT / "data/guard/guard_small.csv")
r = pd.read_csv(ROOT / "data/course/rag_eval.csv").fillna({"gold_section": ""})
qs = [(q, g != "") for q, g in zip(r.question, r.gold_section)] + [(q, False) for q in UNCOVERED]
_, _, test = split(); test = test.groupby("intent").sample(10, random_state=0)
rows = []
MODELS = [("base d1 (no LoRA)", None), ("support LoRA", "runs/d1_lora_support10k_backup"), ("all-decisions LoRA", "runs/d1_lora_all")]
for tag, path in MODELS:
    lora = path is not None
    if lora:
        res = s.d1.load_state_dict(torch.load(ROOT / path / "lora.pt"), strict=False); assert not res.unexpected_keys
    for name, d in (("guard: Gemini v1 (never trained on)", g1), ("guard: hand-made (TRAINING data of all-decisions LoRA)", g2)):
        P = [s.ask({"message": t}, scope_questions(sc), lora=lora) for sc, t in zip(d.scope, d.text)]
        sup, inj, harm = (np.array([a[k]["noul"] for a in P]) for k in ("support", "injection", "harmful"))
        y = d.label.to_numpy(); passed = (inj < 0.1) & (harm < 0.1) & (sup >= 0.5)
        rows.append({"model": tag, "test": name, "correct %": 100 * np.mean(passed == (y == "in_scope")),
                     "legit blocked %": 100 * (1 - passed[y == "in_scope"].mean()), "injection through %": 100 * passed[y == "prompt_injection"].mean(),
                     "harmful through %": 100 * passed[y == "harmful"].mean(), "off-topic through %": 100 * passed[y == "out_of_scope"].mean()})
        print(rows[-1], flush=True)
    p, ok = [], []
    for q, has in qs:
        top, _ = s.retrieve(q)   # embedder top 3 (k=3)
        st = {"customer_message": q, "help_center_sections": [f"{x['title']} > {x['section']}: {x['text']}" for x in top]}
        p.append(s.ask(st, ANS3, lora=lora)["a"]["noul"]); ok.append(has)
    p, ok = np.array(p), np.array(ok)
    rows.append({"model": tag, "test": "answerable check (top 3, one call)", "correct %": 100 * np.mean((p >= 0.3) == ok),
                 "answerable passed %": 100 * np.mean(p[ok] >= 0.3), "uncovered stopped %": 100 * np.mean(p[~ok] < 0.3),
                 "min p answerable": p[ok].min(), "max p uncovered": p[~ok].max()})
    print(rows[-1], flush=True)
    for task, qkey in (("intent", "intent"), ("team", "department"), ("priority", "priority")):
        pr = [s.ask({"message": t}, {qkey: s.lora_questions[qkey]}, lora=lora)[qkey]["choice"] for t in test.text]
        from sklearn.metrics import f1_score
        rows.append({"model": tag, "test": f"{task} (10k test, 400)", "macro-F1": 100 * f1_score(test[qkey], pr, average="macro")})
        print(rows[-1], flush=True)
out = pd.DataFrame(rows).round(1)
out.to_csv(ROOT / "results/lora_everywhere.csv", index=False)
print(out.to_markdown(index=False))
