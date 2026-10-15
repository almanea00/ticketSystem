"""Agent screen (Gradio): paste a ticket -> team, priority, similar past cases, grounded draft reply.

Run: python -m ticketassist.app [--device cuda:1] [--draft-device cuda:0] [--port 7861]
"""

import argparse
import html

import gradio as gr

from .draft import language_of
from .pipeline import GROUNDED_OK, Assistant

EXAMPLES = [
    ["طلبي متأخر من أسبوع والتتبع واقف", "طلبت جهاز من أسبوع والتتبع واقف من ثلاث أيام، ولا أحد يرد علي في الدعم. وش صاير؟"],
    ["Cannot access VPN after password reset",
     "Since resetting my password this morning I cannot connect to the company VPN. It says authentication failed. "
     "I need access for a client meeting at 2pm."],
    ["فاتورة مكررة", "تم خصم مبلغ الاشتراك مرتين هذا الشهر من بطاقتي. أرجو استرداد المبلغ الزائد."],
    ["Data analytics dashboard crashes",
     "Our analytics dashboard crashes whenever we load the quarterly report. Restarting did not help."],
]

CSS = """
.pill{display:inline-block;padding:2px 10px;border-radius:999px;font-size:.85rem;margin:2px 4px 2px 0;border:1px solid #8884}
.high{background:#c0392b22;color:#c0392b}.medium{background:#d68a0022;color:#b07000}.low{background:#2f7d3222;color:#2f7d32}
.case{border:1px solid #8884;border-radius:8px;padding:8px 10px;margin:6px 0}
.case .m{font-size:.8rem;opacity:.75}
.bar{height:6px;border-radius:3px;background:#12706b}
.ok{color:#2f7d32;font-weight:600}.warn{color:#c0392b;font-weight:600}
"""


def triage_html(t):
    probs = sorted(t["queue_probs"].items(), key=lambda x: -x[1])[:3]
    bars = "".join(f"<div style='display:flex;gap:8px;align-items:center'><span style='width:210px'>{html.escape(k)}</span>"
                   f"<div class='bar' style='width:{max(2, int(200 * v))}px'></div><span>{100 * v:.0f}%</span></div>"
                   for k, v in probs)
    esc = t["escalate"]
    return (f"<div><span class='pill'>Team: <b>{html.escape(t['queue'])}</b></span>"
            f"<span class='pill {t['priority']}'>Priority: <b>{t['priority']}</b> ({t['priority_level']:.2f})</span>"
            f"<span class='pill {'high' if esc >= 0.5 else 'low'}'>Needs a person now: {'yes' if esc >= 0.5 else 'no'} "
            f"({100 * esc:.0f}%)</span></div><div style='margin-top:8px'>{bars}</div>")


def cases_html(cases):
    if not cases:
        return "<p><b>No similar past case.</b> Nothing passed the similarity and same-problem checks.</p>"
    out = []
    for c in cases:
        out.append(f"<div class='case'><div><b>[{c['id']}]</b> {html.escape(c['subject'])}</div>"
                   f"<div class='m'>{html.escape(c['queue'])} · cosine {c['cosine']:.2f} · "
                   f"d1 same problem {100 * c['same']:.0f}%</div>"
                   f"<details><summary>ticket and resolution</summary><p>{html.escape(c['body'])}</p>"
                   f"<p><b>Resolution:</b> {html.escape(c['answer'])}</p></details></div>")
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--draft-device", default=None, help="GPU for ALLaM-7B (omit to skip drafting)")
    ap.add_argument("--mode", default="similarity")
    ap.add_argument("--adapter")
    ap.add_argument("--port", type=int, default=7861)
    args = ap.parse_args()
    a = Assistant(args.device, args.draft_device, args.mode, args.adapter)

    def run(subject, body, team_filter):
        if not body.strip():
            return "", "", "", ""
        r = a.handle(subject, body, team_filter=team_filter)
        draft = ""
        if r.draft:
            ok = r.grounded is not None and r.grounded >= GROUNDED_OK
            badge = (f"<span class='ok'>Grounded ✓ ({100 * r.grounded:.0f}%)</span>" if ok else
                     f"<span class='warn'>Check before sending: d1 found unsupported claims ({100 * (r.grounded or 0):.0f}%)</span>")
            rtl = " dir='rtl'" if language_of(r.draft) == "ar" else ""
            draft = f"{badge}<div{rtl} style='white-space:pre-wrap;margin-top:6px'>{html.escape(r.draft)}</div>"
        elif a.drafter is None:
            draft = "<i>Reply drafting is off (start with --draft-device).</i>"
        times = " · ".join(f"{k} {1000 * v:.0f} ms" for k, v in r.timings.items())
        return triage_html(r.triage), cases_html(r.cases), draft, times

    with gr.Blocks(title="Ticket Assistant") as ui:
        gr.Markdown("## Ticket Assistant\nPaste a customer ticket (Arabic or English).")
        with gr.Row():
            with gr.Column(scale=2):
                subject = gr.Textbox(label="Subject")
                body = gr.Textbox(label="Message", lines=6)
                team_filter = gr.Checkbox(label="Only search past tickets of the predicted team", value=False)
                go = gr.Button("Analyse", variant="primary")
                gr.Examples(EXAMPLES, [subject, body])
            with gr.Column(scale=3):
                gr.Markdown("**Triage** (d1‑3B)")
                tri = gr.HTML()
                gr.Markdown("**Draft reply** (ALLaM‑7B, checked by d1)")
                draft = gr.HTML()
        gr.Markdown("**Similar past cases** (EmbeddingGemma cosine → d1 re‑rank)")
        cases = gr.HTML()
        timing = gr.Markdown()
        go.click(run, [subject, body, team_filter], [tri, cases, draft, timing])
        body.submit(run, [subject, body, team_filter], [tri, cases, draft, timing])
    ui.launch(server_name="127.0.0.1", server_port=args.port, css=CSS)


if __name__ == "__main__":
    main()
