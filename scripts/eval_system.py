"""End-to-end evaluation of the ticket assistant (ticketassist/system.py) on the course data (labs/data).

1. Safeguard. Should PASS: the 108 human Arabic course tickets, the 30 answerable rag_eval questions, 200 Bitext English
   tickets. Should be REFUSED: deepset/prompt-injections test (injections and general non-store requests),
   jackhhao jailbreak test (jailbreaks and general prompts), the 2 unanswerable rag_eval questions are reported apart.
   -> false-refusal rate on real customers, block rate on attacks/off-topic.
2. Full pipeline (guard -> intent -> retrieve -> draft -> check) on the 108 course tickets and the 32 rag_eval questions:
   intent accuracy (tickets), right help page retrieved (rag_eval gold section; tickets: the intent's article),
   final action, check problems, time per stage.
3. Failure list for error analysis: results/system_failures.csv (one row per failure with its stage).
Writes results/system_guard.csv, results/system_runs.parquet. Usage: python scripts/eval_system.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ticketassist.system import TicketSystem  # noqa: E402

C = ROOT / "data/course"
# which help-center article covers each intent (as in the course's Lab 2.4)
CAT_DOC = {"ACCOUNT": "account", "CANCEL": "cancellations", "CONTACT": "contact", "DELIVERY": "shipping", "FEEDBACK": "feedback",
           "INVOICE": "invoices", "ORDER": "orders", "PAYMENT": "payments", "REFUND": "refunds", "SHIPPING": "shipping",
           "SUBSCRIPTION": "newsletter"}


def guard_sets():
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    t = pd.read_csv(C / "arabic_tickets.csv")
    r = pd.read_csv(C / "rag_eval.csv").fillna({"gold_section": ""})
    b = pd.read_parquet(ROOT / "data/bitext.parquet")
    b = b[(b.split == "test") & (b.lang == "en")].sample(200, random_state=0)
    d = load_dataset("deepset/prompt-injections", split="test").to_pandas()
    j = pd.read_csv(hf_hub_download("jackhhao/jailbreak-classification", "balanced/jailbreak_dataset_test_balanced.csv", repo_type="dataset"))
    rows = [("course Arabic tickets", x, "pass") for x in t.text]
    rows += [("rag_eval answerable questions", x, "pass") for x in r[r.gold_section != ""].question]
    rows += [("Bitext English tickets", x, "pass") for x in b.text]
    rows += [("deepset: injection", x, "refuse") for x in d[d.label == 1].text]
    rows += [("deepset: benign non-store request", x, "refuse") for x in d[d.label == 0].text]
    rows += [("jackhhao: jailbreak", x[:2000], "refuse") for x in j[j.type == "jailbreak"].prompt]
    rows += [("jackhhao: benign non-store prompt", x[:2000], "refuse") for x in j[j.type == "benign"].prompt]
    rows += [("rag_eval unanswerable", x, "either") for x in r[r.gold_section == ""].question]
    return pd.DataFrame(rows, columns=["set", "text", "expected"])


def main():
    s = TicketSystem()
    # ---- 1. safeguard
    g = guard_sets()
    res = [s.guard(x) for x in g.text]
    g["allowed"] = [r["allowed"] for r in res]
    g["p_support"] = [r["p_support"] for r in res]
    g["p_injection"] = [r["p_injection"] for r in res]
    g["correct"] = np.where(g.expected == "pass", g.allowed, np.where(g.expected == "refuse", ~g.allowed, True))
    summ = g.groupby(["set", "expected"]).agg(n=("text", "size"), passed=("allowed", "mean"), correct=("correct", "mean")).reset_index()
    summ["passed"] = (100 * summ.passed).round(1)
    summ["correct"] = (100 * summ.correct).round(1)
    g.to_csv(ROOT / "results/system_guard_items.csv", index=False)
    summ.to_csv(ROOT / "results/system_guard.csv", index=False)
    print(summ.to_markdown(index=False), flush=True)
    pos, neg = g[g.expected == "pass"], g[g.expected == "refuse"]
    print(f"\nfalse refusals of real customers: {100 * (~pos.allowed).mean():.1f}% ({(~pos.allowed).sum()}/{len(pos)}) | "
          f"attacks / off-topic blocked: {100 * (~neg.allowed).mean():.1f}% ({(~neg.allowed).sum()}/{len(neg)})", flush=True)

    # ---- 2. full pipeline
    b = pd.read_parquet(ROOT / "data/bitext.parquet")
    cat = dict(zip(b.intent, b.category))
    t = pd.read_csv(C / "arabic_tickets.csv")
    r = pd.read_csv(C / "rag_eval.csv").fillna({"gold_section": ""})
    items = [{"source": f"course ticket ({d})", "text": x, "intent": i, "gold_doc": CAT_DOC[cat[i]], "gold_section": None}
             for x, i, d in zip(t.text, t.intent, t.dialect)]
    items += [{"source": f"rag_eval ({l})", "text": q, "intent": None, "gold_doc": None, "gold_section": gs or None}
              for q, l, gs in zip(r.question, r.lang, r.gold_section)]
    runs = []
    for n, it in enumerate(items):
        o = s.handle(it["text"])
        secs = [x["id"] for x in o.get("sections", [])]
        row = {**it, "action": o["action"], "reply": o.get("reply", ""), "guard_allowed": o["guard"]["allowed"],
               "pred_intent": o.get("intent", {}).get("intent"), "intent_conf": o.get("intent", {}).get("confidence"),
               "sections": " | ".join(secs), "top_p_answers": o["sections"][0]["p_answers"] if o.get("sections") else None,
               "problems": " ; ".join(o.get("check", {}).get("problems", [])),
               **{f"t_{k}": v for k, v in o["timings"].items()}}
        row["intent_ok"] = None if it["intent"] is None else row["pred_intent"] == it["intent"]
        row["retrieval_ok"] = (it["gold_section"] in secs) if it["gold_section"] else ((it["gold_doc"] in {x.split("#")[0] for x in secs}) if it["gold_doc"] else None)
        runs.append(row)
        if n % 20 == 0:
            print(n, len(items), flush=True)
    df = pd.DataFrame(runs)
    df.to_parquet(ROOT / "results/system_runs.parquet")

    def stage(x):  # first failing stage, in pipeline order (course Lab 2.4)
        if x.action.startswith("refused"):
            return "guard (refused a real customer)" if x.gold_section is not None or x.intent is not None else "guard (refused)"
        if x.intent_ok is False:
            return "classification"
        if x.retrieval_ok is False:
            return "retrieval"
        if x.action == "human review":
            return "generation / checks"
        return "ok"
    df["failure_stage"] = df.apply(stage, axis=1)
    df[df.failure_stage != "ok"].to_csv(ROOT / "results/system_failures.csv", index=False)
    by = df.groupby("source").agg(n=("text", "size"), refused=("action", lambda a: 100 * a.str.startswith("refused").mean()),
                                  intent_acc=("intent_ok", lambda a: 100 * a.dropna().astype(float).mean() if a.notna().any() else np.nan),
                                  retrieval_ok=("retrieval_ok", lambda a: 100 * a.dropna().astype(float).mean() if a.notna().any() else np.nan),
                                  ready_to_send=("action", lambda a: 100 * (a == "send (agent approves)").mean()),
                                  human_review=("action", lambda a: 100 * (a == "human review").mean()),
                                  p50_total_s=("t_total", "median")).round(1).reset_index()
    print(by.to_markdown(index=False))
    print("\nfailure stage counts:", df.failure_stage.value_counts().to_dict())
    print("check problems:", df.problems.str.split(" ; ").explode().replace("", np.nan).dropna().str.replace(r"\(.*\)", "", regex=True).str.strip().value_counts().to_dict())
    print("\ntime per stage (median s):", df[[c for c in df if c.startswith("t_")]].median().round(2).to_dict())


if __name__ == "__main__":
    main()
