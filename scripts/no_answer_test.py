"""Does ALLaM invent answers when the help center has none? Gemma alone (top 3 by cosine -> ALLaM) vs Gemma + d1
(d1 judges the top 3; best < 0.3 -> human review, no reply). Questions: 20 store questions the course KB does not cover
(written for this test; checked against the KB headings), the 2 unanswerable rag_eval questions, and 10 answerable
rag_eval questions as a control. Replies are saved for reading: results/no_answer_test.csv."""
import sys, time
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ticketassist.system import TicketSystem, language  # noqa: E402

UNCOVERED = [
    "Do you sell gaming laptops with an RTX 5090?", "What is the warranty period on Samsung TVs?",
    "Do you have a physical store in Jeddah I can visit?", "Can I buy a gift card for my friend?",
    "Do you offer installation for air conditioners?", "Can I trade in my old iPhone for credit?",
    "How many loyalty points do I get per riyal?", "Do you price match if Amazon is cheaper?",
    "Are you hiring? How do I apply for a job?", "Is the PS5 Pro back in stock?",
    "عندكم فرع في الرياض اقدر ازوره؟", "كم مدة الضمان على الثلاجات؟", "تبيعون بطاقات هدايا؟",
    "هل توفرون خدمة تركيب المكيفات؟", "ابي ابدل جوالي القديم بخصم على الجديد، ينفع؟",
    "وش نظام نقاط الولاء عندكم؟", "عندكم آيفون 17 برو ماكس متوفر؟", "تطابقون السعر لو لقيته ارخص بمكان ثاني؟",
    "ابي اتوظف عندكم وش الطريقة؟", "هل الشاحن الأصلي يجي مع الجوال؟",
]


def main():
    s = TicketSystem()
    r = pd.read_csv(ROOT / "data/course/rag_eval.csv").fillna({"gold_section": ""})
    items = [(q, "uncovered (written)") for q in UNCOVERED] + [(q, "uncovered (rag_eval)") for q in r[r.gold_section == ""].question]
    items += [(q, "answerable control") for q in r[r.gold_section != ""].groupby("lang").head(5).question]
    rows = []
    for q, kind in items:
        lang = language(q)
        top, cand = s.retrieve(q)
        by_cos = sorted(cand, key=lambda x: x["cosine_rank"])[:3]
        t = time.time(); a = s.draft(q, by_cos, lang); ta = time.time() - t                 # Gemma alone
        best = top[0]["p_answers"]
        if best < 0.3:
            b, tb = "(no reply: human review)", 0.0                                            # Gemma + d1
        else:
            t = time.time(); b = s.draft(q, top, lang); tb = time.time() - t
        rows.append({"kind": kind, "question": q, "sections": " | ".join(x["id"] for x in by_cos), "d1_best": round(best, 3),
                     "gemma_alone_reply": a, "gemma_d1_reply": b, "s_gemma_alone": round(ta, 1), "s_gemma_d1": round(tb, 1)})
        print(len(rows), kind, q[:50], "| d1", round(best, 2), flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "results/no_answer_test.csv", index=False)


if __name__ == "__main__":
    main()
