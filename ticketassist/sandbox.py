"""Sandbox: watch d1-3B decide on labelled messages it was never trained on.

Pick a dataset (Banking77: 77 banking intents, Bitext: 27 support intents) and a language, draw a random
labelled message (or type your own), and see d1's top choices with probabilities next to the true label.
The "Custom question" tab takes any state / question / options as JSON.

Run: python -m ticketassist.sandbox [--device cuda:1] [--port 7862]
"""

import argparse
import html
import json
import random
from pathlib import Path

import gradio as gr
import pandas as pd
import torch
from transformers import AutoModel

from .decide import MODEL

ROOT = Path(__file__).resolve().parent.parent
LANGS = {"English": "en", "فصحى": "ar", "Najdi": "najdi"}
CSS = """
.bar{height:10px;border-radius:5px;background:#12706b}.bar.hit{background:#2f7d32}
.row{display:grid;grid-template-columns:minmax(0,260px) 1fr 52px;gap:10px;align-items:center;margin:4px 0;font-size:.92rem}
.ok{color:#2f7d32;font-weight:700}.no{color:#c0392b;font-weight:700}.msg{font-size:1.15rem;padding:8px 0}
"""


def nice(label):
    return label.replace("_", " ").capitalize()


def load_sets():
    sets = {}
    b = pd.read_parquet(ROOT / "data/banking77_showcase.parquet")
    sets["Banking77 (77 intents)"] = (b.rename(columns={"label_text": "label"}), sorted(b["label_text"].unique()),
                                      "What does the bank customer want?")
    t = pd.read_parquet(ROOT / "data/bitext.parquet")
    t = t[t["split"] == "test"].rename(columns={"intent": "label"})
    sets["Bitext support (27 intents)"] = (t, sorted(t["label"].unique()), "What does the customer want to do?")
    return sets


def bars(probs, truth=None, k=5):
    top = sorted(probs.items(), key=lambda x: -x[1])[:k]
    out = []
    for name, p in top:
        hit = " hit" if name == truth else ""
        out.append(f"<div class='row'><span>{html.escape(nice(name))}{' ✓' if hit else ''}</span>"
                   f"<div class='bar{hit}' style='width:{max(1, int(100 * p))}%'></div><span>{100 * p:.0f}%</span></div>")
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--port", type=int, default=7862)
    args = ap.parse_args()
    sets = load_sets()
    model = AutoModel.from_pretrained(MODEL, trust_remote_code=True, dtype=torch.bfloat16,
                                      device_map={"": args.device}).eval()

    @torch.no_grad()
    def ask(state, questions):
        return model.system_one(state, questions)["answers"]

    def draw(name, lang):
        df, _, _ = sets[name]
        g = df[df["lang"] == LANGS[lang]]
        r = g.iloc[random.randrange(len(g))]
        return r["text"], r["label"]

    def decide(name, text, truth, score):
        if not text.strip():
            return "", "", score, ""
        df, labels, instr = sets[name]
        q = {"intent": {"type": "choice", "instructions": instr, "criteria": {l: nice(l) for l in labels}}}
        a = ask({"message": text}, q)["intent"]
        verdict = ""
        if truth:
            ok = a["choice"] == truth
            score = [score[0] + ok, score[1] + 1]
            verdict = (f"<span class='{'ok' if ok else 'no'}'>{'Correct' if ok else 'Wrong'}</span> · "
                       f"true label: <b>{html.escape(nice(truth))}</b> · d1 chose <b>{html.escape(nice(a['choice']))}</b> "
                       f"({100 * a['confidence']:.0f}% sure)")
        else:
            verdict = f"d1 chose <b>{html.escape(nice(a['choice']))}</b> ({100 * a['confidence']:.0f}% sure) · no true label (typed by you)"
        rtl = " dir='rtl'" if any("؀" <= c <= "ۿ" for c in text) else ""
        head = f"<div class='msg'{rtl}>{html.escape(text)}</div>{verdict}"
        tally = f"Session score: {score[0]} / {score[1]} correct" if score[1] else ""
        return head, bars(a["probabilities"], truth), score, tally

    def custom(state_json, q_json):
        try:
            state, q = json.loads(state_json), json.loads(q_json)
            ans = ask(state, q)
        except Exception as e:  # noqa: BLE001 - show the problem in the page
            return f"<span class='no'>{html.escape(type(e).__name__)}: {html.escape(str(e))}</span>"
        parts = []
        for name, a in ans.items():
            if "probabilities" in a:
                parts.append(f"<h4>{html.escape(name)} ({a['type']})</h4>" + bars(a["probabilities"], k=10))
            else:
                parts.append(f"<h4>{html.escape(name)} (yes/no)</h4>" + bars({"yes": a["noul"], "no": 1 - a["noul"]}))
        return "".join(parts)

    with gr.Blocks(title="d1 Sandbox") as ui:
        gr.Markdown("## d1‑3B sandbox\nZero‑shot: d1 was never trained on these datasets. It only sees the list of intent names.")
        with gr.Tab("Labelled messages"):
            score = gr.State([0, 0])
            truth = gr.State("")
            with gr.Row():
                name = gr.Dropdown(list(sets), value=list(sets)[0], label="Dataset")
                lang = gr.Radio(list(LANGS), value="English", label="Language")
                nxt = gr.Button("Random message", variant="primary")
            text = gr.Textbox(label="Message (edit it or type your own, then press Decide)", lines=2)
            go = gr.Button("Decide")
            head, top = gr.HTML(), gr.HTML()
            tally = gr.Markdown()

            def new(name, lang, score):
                t, y = draw(name, lang)
                h, b, s, ta = decide(name, t, y, score)
                return t, y, h, b, s, ta

            nxt.click(new, [name, lang, score], [text, truth, head, top, score, tally])
            go.click(lambda n, t, y, s: decide(n, t, y, s), [name, text, truth, score], [head, top, score, tally])
            text.input(lambda: "", None, truth)  # once edited, the message has no true label
        with gr.Tab("Custom question"):
            st = gr.Code(json.dumps({"message": "والله إن ما رجعتوا فلوسي اليوم لأشتكي عليكم في وزارة التجارة!"}, ensure_ascii=False, indent=1),
                         language="json", label="State")
            qq = gr.Code(json.dumps({"escalate": {"type": "noul", "instructions": "Does a person need to act right away?"},
                                     "team": {"type": "choice", "instructions": "Which team?",
                                              "criteria": {"billing": "Charges and refunds", "technical": "App faults",
                                                           "orders": "Orders and delivery"}},
                                     "urgency": {"type": "score", "instructions": "How urgent?",
                                                 "criteria": ["Can wait", "Today", "Now"]}}, indent=1),
                         language="json", label="Questions")
            run = gr.Button("Ask d1", variant="primary")
            res = gr.HTML()
            run.click(custom, [st, qq], res)
    ui.launch(server_name="127.0.0.1", server_port=args.port, css=CSS)


if __name__ == "__main__":
    main()
