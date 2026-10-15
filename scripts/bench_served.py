"""Re-run the model card's benchmarks on the served system (.env: d1 + merged LoRA on vLLM, ALLaM on vLLM).
  decisions  intent / team / priority macro-F1 on the full 2,000-message test set and on the 400-message sample
             (10 per intent) the card's step table uses
  safeguard  Gemini v1, the same 800 rows (200 per label, random_state=1): grid of cut-offs, and the chosen 0.2 / 0.05
  RAG        Hit@1 / Hit@k / MRR on rag_eval (dense EmbeddingGemma); the answerable pre-check on the 30 answerable
             and the 22 uncovered questions
  latency    the full pipeline on 80 messages (60 Arabic support test messages, 20 English rag_eval questions)
  replies    the 22 uncovered questions through the full pipeline, replies saved for reading
-> results/served_bench.json, results/served_replies.csv"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.no_answer_test import UNCOVERED  # noqa: E402
from scripts.support10k_bench import split  # noqa: E402
from ticketassist.config import CONFIG  # noqa: E402
from ticketassist.guard import scope_questions  # noqa: E402
from ticketassist.system import TicketSystem  # noqa: E402

out = {"setup": CONFIG.describe(), "date": time.strftime("%Y-%m-%d")}
s = TicketSystem()
print(out["setup"], flush=True)

# ---- decisions
_, _, test = split()
t = time.time()
res = s.d1.system_one_batch([({"message": m}, s.decision_questions) for m in test.text])
secs = time.time() - t
pred = {k: [r["answers"][k]["choice"] for r in res] for k in ("intent", "department", "priority")}
sample = test.groupby("intent").sample(10, random_state=0).index
dec = {}
for k in pred:
    p = np.array(pred[k])
    dec[k] = {"macro_f1_2000": 100 * f1_score(test[k], p, average="macro"),
              "macro_f1_400_sample": 100 * f1_score(test[k].loc[sample], p[sample], average="macro")}
dec["throughput_messages_per_s"] = len(test) / secs
out["decisions"] = dec
print("decisions", dec, flush=True)

# ---- safeguard
g = pd.read_csv(ROOT / "data/guard/guardrail_decision_dataset_5000.csv").groupby("label").sample(200, random_state=1)
P = s.d1.system_one_batch([({"message": t}, scope_questions(sc)) for sc, t in zip(g.scope, g.text)])
sup, inj, harm = (np.array([a["answers"][k]["noul"] for a in P]) for k in ("support", "injection", "harmful"))
y = g.label.to_numpy(); L = y == "in_scope"
grid = []
for bt in (0.1, 0.2, 0.3, 0.5):
    for st in (0.02, 0.05, 0.1, 0.2, 0.3, 0.5):
        ok = (inj < bt) & (harm < bt) & (sup >= st)
        grid.append({"block": bt, "support": st, "correct": 100 * np.mean(ok == L), "legit_blocked": 100 * (1 - ok[L].mean()),
                     "injection_through": 100 * ok[y == "prompt_injection"].mean(), "harmful_through": 100 * ok[y == "harmful"].mean(),
                     "offtopic_through": 100 * ok[y == "out_of_scope"].mean()})
out["safeguard_grid"] = grid
out["safeguard_chosen"] = [r for r in grid if r["block"] == CONFIG.guard_block and r["support"] == CONFIG.guard_support][0]
print("safeguard", out["safeguard_chosen"], flush=True)

# ---- RAG
r = pd.read_csv(ROOT / "data/course/rag_eval.csv").fillna({"gold_section": ""})
ids = [x["id"] for x in s.kb]; rows = []
for q, gold, lang in zip(r.question, r.gold_section, r.lang):
    order = list(np.argsort(-(s.kb_vecs @ s.embedder.query(q))))
    sec, _ = s.retrieve(q)
    rows.append({"lang": lang, "gold": gold, "rank": [ids[i] for i in order].index(gold) + 1 if gold else None, "p": sec[0]["p_answers"]})
d = pd.DataFrame(rows); a = d[d.gold != ""]
unc = [s.retrieve(q)[0][0]["p_answers"] for q in UNCOVERED] + list(d[d.gold == ""].p)
k = CONFIG.top_k
out["rag"] = {"hit1": float((a["rank"] == 1).mean()), f"hit{k}": float((a["rank"] <= k).mean()), "mrr": float((1 / a["rank"]).mean()),
              f"hit{k}_arabic": float((a[a.lang == "ar"]["rank"] <= k).mean()),
              "answerable_passed": int((a.p >= CONFIG.answer_min).sum()), "answerable_n": len(a),
              "uncovered_stopped": int((np.array(unc) < CONFIG.answer_min).sum()), "uncovered_n": len(unc)}
print("rag", out["rag"], flush=True)

# ---- latency + replies
msgs = list(test.sample(60, random_state=5).text) + list(r[(r.lang == "en") & (r.gold_section != "")].question[:20])
T = {}
for m in msgs:
    o = s.handle(m)
    for key, v in o["timings"].items():
        T.setdefault(key, []).append(1000 * v)
out["latency_ms"] = {key: {"n": len(v), "p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95))} for key, v in T.items()}
print("latency", out["latency_ms"], flush=True)
rep = []
for q in UNCOVERED + list(r[r.gold_section == ""].question):
    o = s.handle(q)
    rep.append({"question": q, "action": o["action"], "p_answers": (o.get("sections") or [{}])[0].get("p_answers"), "reply": o.get("reply", "")})
pd.DataFrame(rep).to_csv(ROOT / "results/served_replies.csv", index=False)
out["uncovered_actions"] = pd.Series([x["action"] for x in rep]).value_counts().to_dict()
json.dump(out, open(ROOT / "results/served_bench.json", "w"), indent=1, ensure_ascii=False)
print("done", flush=True)
