"""Injection-first guard: raw yes-probabilities of the scope guard questions on a guard CSV, then the pass/block
trade-off at several blocking thresholds (block if p_injection >= t or p_harmful >= t, else in_scope needs support >= 0.5).
Usage: python scripts/guard_threshold.py data/guard/guard_small.csv [modes] [max_rows_per_label]"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ticketassist.guard import scope_questions, scope_questions_fs, scope_questions_fsv2  # noqa: E402
from ticketassist.system import TicketSystem  # noqa: E402

f = sys.argv[1]; modes = (sys.argv[2] if len(sys.argv) > 2 else "scope_noul,scope_noul_fs").split(",")
d = pd.read_csv(ROOT / f)
if len(sys.argv) > 3:
    d = d.groupby("label").sample(int(sys.argv[3]), random_state=0).reset_index(drop=True)
s = TicketSystem(draft=False)
Q = {"scope_noul": scope_questions, "scope_noul_fs": scope_questions_fs, "scope_noul_fsv2": scope_questions_fsv2}
for m in modes:
    P = [s.ask({"message": t}, Q[m](sc)) for sc, t in zip(d.scope, d.text)]
    sup, inj, harm = (np.array([a[k]["noul"] for a in P]) for k in ("support", "injection", "harmful"))
    print(f"\n{m} on {f} ({len(d)} rows)")
    print(f"{'block at':>9} | {'correct':>7} | {'legit blocked':>13} | {'injection through':>17} | {'harmful through':>15} | {'off-topic through':>17}")
    for t in (0.5, 0.3, 0.2, 0.1, 0.05):
        passed = (inj < t) & (harm < t) & (sup >= 0.5)
        y = d.label.to_numpy()
        th = lambda l: f"{100 * passed[y == l].mean():.1f}%"  # noqa: E731
        print(f"{t:>9} | {100 * np.mean(passed == (y == 'in_scope')):6.1f}% | {th('in_scope').replace('%','')} -> {100*(1-passed[y=='in_scope'].mean()):.1f}% | {th('prompt_injection'):>17} | {th('harmful'):>15} | {th('out_of_scope'):>17}")
