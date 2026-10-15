"""Agent screen for the ticket assistant: paste a ticket, see every stage's decision, the draft reply and why.

Run: python -m ticketassist.web [--port 7861]
"""

import argparse
import html

import gradio as gr

from .system import TicketSystem

ACTION = {"reply ready": ("ok", "Reply ready"), "asked for details": ("warn", "Asked the customer for details"), "working": ("warn", "Working…"), "triaging": ("warn", "Screening…"), "waiting": ("warn", "Waiting"), "writing reply…": ("warn", "Writing reply…"), "send (agent approves)": ("ok", "Ready to send · agent approves"), "human review": ("warn", "Human review"),
          "refused: out of scope": ("warn", "Out of scope"),
          "refused: prompt injection": ("no", "Blocked: prompt injection"), "refused: harmful": ("no", "Blocked: harmful request"), "classified": ("ok", "Classified")}
CSS = """
.gradio-container{max-width:100%!important;padding:6px 14px!important}footer{display:none!important}
.gradio-container h2{margin:0!important;font-size:1.1rem!important}
.badge{display:inline-block;padding:2px 10px;border-radius:999px;font-weight:600;font-size:.9rem}
.badge.ok{background:#2f7d3222;color:#2f7d32}.badge.warn{background:#d68a0022;color:#a5620f}.badge.no{background:#c0392b22;color:#c0392b}
.sgrid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;font-size:.8rem}
.stage{border:1px solid #8884;border-radius:8px;padding:6px 9px;min-width:0}.stage.w2{grid-column:span 2}
.stage h4{margin:0 0 4px 0;font-size:.82rem}.muted{opacity:.7;font-size:.75rem}
.bar{height:6px;border-radius:3px;background:#12706b;display:inline-block;vertical-align:middle}
.reply{white-space:pre-wrap;font-size:.88rem;padding:6px 9px;border-left:4px solid #8c2f5f;background:#8c2f5f14;border-radius:0 6px 6px 0;max-height:260px;overflow:auto}
.dgrid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}.dlab{font-weight:600;margin-bottom:2px}
.prob{display:grid;grid-template-columns:minmax(0,1fr) 70px 34px;gap:6px;align-items:center;line-height:1.35}
.prob span:first-child{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.secs{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:6px}
.sec{border:1px solid #8883;border-radius:6px;padding:4px 7px}.sec b{display:block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.sectext{display:-webkit-box;-webkit-line-clamp:4;-webkit-box-orient:vertical;overflow:hidden;margin-top:2px}
.sec summary{cursor:pointer;list-style:none}.sec[open] summary .sectext{display:none}.sec .sectext.full{max-width:none;margin:2px 0}
.stage ul{margin:2px 0 0 16px;padding:0}.stage.w3{grid-column:1/-1}
.prompts{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:6px;margin-top:6px}
.prompt summary{cursor:pointer;font-weight:600}.prompt pre{white-space:pre-wrap;font-size:.75rem;background:#8881;padding:6px;border-radius:6px;max-height:320px;overflow:auto;margin:4px 0}
.queue{border-collapse:collapse;width:100%;font-size:.8rem;margin-top:6px}.queue td,.queue th{padding:4px 7px;border-bottom:1px solid #8883;text-align:left;vertical-align:top}
.qreply{max-width:460px}.qsec summary{cursor:pointer;white-space:nowrap}.qsec{margin:1px 0}
.sectext.full{display:block;-webkit-line-clamp:unset;white-space:pre-wrap;max-width:360px;margin:2px 0 4px 12px}
"""


def prob_row(label, p):
    return (f"<div class='prob'><span title='{html.escape(label)}'>{html.escape(label)}</span>"
            f"<span><span class='bar' style='width:{max(2, int(70 * p))}px'></span></span><span>{100 * p:.0f}%</span></div>")


def render(r):
    cls, label = ACTION[r["action"]]
    rtl = " dir='rtl'" if r["language"] == "ar" else ""
    head = f"<span class='badge {cls}'>{label}</span>"
    reply = f"<div class='reply'{rtl}>{html.escape(r.get('reply') or '(no reply drafted)')}</div>"
    g = r["guard"]
    st = [f"<div class='stage'><h4>① Safeguard · {'passed' if g['allowed'] else 'blocked: ' + html.escape(g['reason'])}</h4>"
          + prob_row("support request (pass ≥ 5%)", g["p_support"]) + prob_row("prompt injection (block ≥ 20%)", g["p_injection"])
          + prob_row("harmful (block ≥ 20%)", g["p_harmful"] or 0) + "</div>"]
    if "intent" in r:
        i = r["intent"]
        pcls = {"high": "no", "medium": "warn", "low": "ok"}[i["priority"]]
        st.append("<div class='stage w2'><h4>② Decisions · decision model</h4><div class='dgrid'>"
                  "<div><div class='dlab'>Intent (40)</div>" + "".join(prob_row(k, p) for k, p in i["top3"]) + "</div>"
                  "<div><div class='dlab'>Team (13)</div>" + "".join(prob_row(k.replace('_', ' '), p) for k, p in i["team_top3"]) + "</div>"
                  f"<div><div class='dlab'>Priority <span class='badge {pcls}'>{i['priority'].capitalize()}</span></div>"
                  + "".join(prob_row(lv.capitalize(), i["priority_probs"][lv]) for lv in ("high", "medium", "low")) + "</div></div></div>")
    if r.get("sections"):
        cards = "".join(f"<details class='sec'><summary><b>[{n}] {html.escape(x['title'])} › {html.escape(x['section'])}</b>"
                        f"<span class='muted'>cosine {x['cosine']:.2f} · <code>{html.escape(x['id'])}</code></span>"
                        f"<div class='sectext'>{html.escape(x['text'])}</div></summary><div class='sectext full'>{html.escape(x['text'])}</div></details>"
                        for n, x in enumerate(r["sections"], 1))
        sq = f" · searched with: <i>{html.escape(r.get('search_query') or '')}</i>" if r.get("search_query") else ""
        st.append(f"<div class='stage w3'><h4>③ Help-center sections · EmbeddingGemma top 4 (the reply answers from these){sq}</h4><div class='secs'>" + cards + "</div></div>")
    st.append(prompts_panel(r))
    tm = " · ".join(f"{k} {v * 1000:.0f} ms" if v < 1 else f"{k} {v:.1f} s" for k, v in r["timings"].items())
    return head, reply, "<div class='sgrid'>" + "".join(st) + "</div>", f"<span class='muted'>{tm}</span>"


def prompts_panel(r):
    """Everything each model was given for this ticket: the reply writer's system prompt and user message, and the
    state + questions of every d1 call (d1 answers by option probabilities, so its "prompt" is a state and questions)."""
    import json
    parts = []
    for m in r.get("llm_messages") or []:
        parts.append(f"<details class='prompt'><summary>Reply writer · {html.escape(m['role'])} "
                     f"{'prompt' if m['role'] == 'system' else 'message'}</summary><pre>{html.escape(m['content'])}</pre></details>")
    for name, p in (r.get("d1_prompts") or {}).items():
        body = json.dumps(p, ensure_ascii=False, indent=1)
        parts.append(f"<details class='prompt'><summary>decision model · {html.escape(name)}: state and questions</summary>"
                     f"<pre>{html.escape(body)}</pre></details>")
    if not parts:
        return ""
    return ("<div class='stage w3'><details><summary><h4 style='display:inline'>Prompts: exactly what each model was given "
            f"({len(parts)})</h4></summary><div class='prompts'>" + "".join(parts) + "</div></details></div>")


RANK = {"high": 0, "medium": 1, "low": 2}


def queue_table(items):
    """items: dicts with text, triage, order, status, reply -> HTML table in processing order, plus a per-team count."""
    rows = []
    for it in items:
        t = it["triage"]
        if t["intent"]:
            i = t["intent"]
            pr = f"<span class='badge {dict(high='no', medium='warn', low='ok')[i['priority']]}'>{i['priority'].capitalize()}</span>"
            team, intent = html.escape(i["team"].replace("_", " ")), html.escape(i["intent"])
        elif it["status"] == "triaging":
            pr, team, intent = "…", "", ""
        else:
            pr, team, intent = "<span class='badge no'>Blocked</span>", "—", html.escape(t["guard"]["reason"] or "")
        cls, lab = ACTION.get(it["status"], ("warn", it["status"]))
        rtl = " dir='rtl'" if any("\u0600" <= ch <= "\u06ff" for ch in it["text"]) else ""
        why = f"<div class='muted'>{html.escape(it['why'])}</div>" if it.get("why") else ""
        reply = html.escape(it.get("reply") or "") or ""
        srcs = "".join(f"<details class='qsec'><summary>[{n}] {html.escape(x['title'])} › {html.escape(x['section'])} "
                       f"<span class='muted'>{x['cosine']:.2f}</span></summary><div class='sectext full'>{html.escape(x['text'])}</div></details>"
                       for n, x in enumerate(it.get("sections") or [], 1))
        rows.append(f"<tr><td>{it['order']}</td><td>{pr}</td><td>{team}</td><td>{intent}</td><td{rtl}>{html.escape(it['text'])}</td>"
                    f"<td><span class='badge {cls}'>{lab}</span>{why}</td><td>{srcs}</td><td{rtl} class='qreply'>{reply}</td></tr>")
    teams = {}
    for it in items:
        if it["triage"]["intent"]:
            k = it["triage"]["intent"]["team"].replace("_", " ")
            teams[k] = teams.get(k, 0) + 1
    inbox = " · ".join(f"<b>{html.escape(k)}</b> {v}" for k, v in sorted(teams.items(), key=lambda x: -x[1]))
    return (f"<div class='muted'>Team inboxes: {inbox or '—'}</div><table class='queue'><tr><th>#</th><th>Priority</th><th>Team</th>"
            "<th>Intent</th><th>Ticket</th><th>Status</th><th>Retrieved sections (click to read)</th><th>Reply</th></tr>" + "".join(rows) + "</table>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7861)
    args = ap.parse_args()
    s = TicketSystem()
    s.writer  # load the reply writer now, not on the first ticket
    s.handle("warm up: where is my order?")

    def run(text):
        """Streams: the page updates after every stage and while the reply is being written."""
        if not text.strip():
            yield "", "", "", ""
            return
        for out in s.handle_stream(text.strip()):
            yield render(out)

    def run_queue(text):
        """Streams too: each ticket appears as soon as it is screened (the list re-sorts by priority as they come),
        then the replies are written high priority first and appear word by word."""
        lines = [x.strip() for x in text.splitlines() if x.strip()]
        items = [{"text": x, "triage": {"guard": {"reason": ""}, "intent": None}, "n": n, "status": "triaging", "order": ""} for n, x in enumerate(lines)]

        def ordered():
            key = lambda it: (2 if it["status"] == "triaging" else 0 if it["triage"]["intent"] else 1,  # noqa: E731
                              RANK[it["triage"]["intent"]["priority"]] if it["triage"]["intent"] else 9, it["n"])
            items.sort(key=key)
            for k, it in enumerate(items, 1):
                it["order"] = k
            return queue_table(items)

        yield ordered()
        for it in sorted(items, key=lambda it: it["n"]):
            it["triage"] = s.triage(it["text"])
            it["status"] = "waiting" if it["triage"]["intent"] else "refused: " + it["triage"]["guard"]["label"].replace("_", " ")
            yield ordered()
        for it in list(items):
            if it["status"] != "waiting":
                continue
            it["status"] = "writing reply…"
            for out in s.handle_stream(it["text"]):
                it["reply"] = out.get("reply", "")
                it["sections"] = out.get("sections") or []
                yield queue_table(items)
            it["status"] = out["action"]
            it["why"] = "; ".join((out.get("check") or {}).get("problems") or [])
            yield queue_table(items)

    with gr.Blocks(title="Ticket Assistant") as ui:
        gr.Markdown(f"## Ticket Assistant <span style='font-size:.8rem;opacity:.7;font-weight:400'>· {html.escape(s.cfg.describe())} </span>")
        with gr.Tab("One ticket"):
          with gr.Row(equal_height=False):
            with gr.Column(scale=2, min_width=300):
                box = gr.Textbox(show_label=False, lines=3, placeholder="اكتب رسالة العميل… / paste the customer's message…")
                go = gr.Button("Handle ticket", variant="primary", size="sm")
                badge = gr.HTML()
                reply = gr.HTML()
                timing = gr.HTML()
            with gr.Column(scale=5):
                stages = gr.HTML()
        with gr.Tab("Queue (many tickets, answered in priority order)"):
            qbox = gr.Textbox(show_label=False, lines=6, placeholder="One ticket per line. Every ticket is screened and classified first, then answered high priority first.")
            qgo = gr.Button("Triage and answer in priority order", variant="primary", size="sm")
            qout = gr.HTML()
            qgo.click(run_queue, qbox, qout)
        go.click(run, box, [badge, reply, stages, timing])
        box.submit(run, box, [badge, reply, stages, timing])
    ui.launch(server_name="127.0.0.1", server_port=args.port, css=CSS)


if __name__ == "__main__":
    main()
