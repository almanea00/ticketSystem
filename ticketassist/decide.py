"""LiquidAI d1-3B as the decision model: typed answers in one forward pass, no generated text.

  route      choice  -> which of the 10 teams (queue)
  priority   score   -> low / medium / high
  escalate   noul    -> does a person need to act right away?
  same       noul    -> is a past ticket the same problem as the new one? (re-ranking retrieved tickets)
  grounded   noul    -> is every claim in a draft reply supported by the cited past tickets?
"""

import torch
from transformers import AutoModel

MODEL = "LiquidAI/d1-3B"

QUEUES = {
    "Technical Support": "Faults and errors in software, platforms, integrations, data or security incidents",
    "Product Support": "Questions about how a product or feature works, setup, configuration, product defects",
    "Customer Service": "Orders, accounts, general service complaints and requests from customers",
    "IT Support": "Internal IT: devices, laptops, network, access, passwords, office software",
    "Billing and Payments": "Invoices, charges, refunds, payment methods, subscriptions",
    "Returns and Exchanges": "Returning or exchanging a purchased product",
    "Service Outages and Maintenance": "A service is down or degraded, scheduled maintenance, outages",
    "Sales and Pre-Sales": "Pricing, quotes, product information before buying, demos",
    "Human Resources": "Employment, payroll, leave, hiring, staff matters",
    "General Inquiry": "Anything else, general questions",
}
PRIORITIES = ["low", "medium", "high"]


def ticket_state(subject: str, body: str) -> dict:
    return {"subject": subject, "message": body}


QUESTIONS = {
    "route": {"type": "choice", "instructions": "Which support team should handle this ticket?", "criteria": QUEUES},
    "priority": {"type": "score", "instructions": "How urgent is this ticket for the business?",
                 "criteria": ["Low: can wait, minor question or request",
                              "Medium: affects work but there is a workaround",
                              "High: blocking work, outage, security or data loss"]},
    "escalate": {"type": "noul", "instructions": "Does a person need to act on this ticket right away "
                 "(security breach, outage, legal threat, very angry customer)?"},
}
SAME = {"type": "noul", "instructions": "Is the past ticket about the same problem as the new ticket, "
        "so that its resolution would help answer the new ticket?"}
GROUNDED = {"type": "noul", "instructions": "Is every factual claim and promised action in the draft reply "
            "supported by the past tickets and their resolutions?"}


class Decider:
    def __init__(self, device: str = "cuda"):
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        # device_map loads the weights straight onto the GPU (no 6.5 GB copy in RAM first)
        self.model = AutoModel.from_pretrained(MODEL, trust_remote_code=True, dtype=dtype, device_map={"": device}).eval()

    @torch.no_grad()
    def triage(self, tickets: list[tuple[str, str]], bs: int = 32) -> list[dict]:
        """[(subject, body)] -> [{queue, queue_conf, queue_probs, priority, priority_level, escalate}]"""
        out = []
        for s in range(0, len(tickets), bs):
            res = self.model.system_one_batch([(ticket_state(*t), QUESTIONS) for t in tickets[s:s + bs]])
            for r in res:
                a = r["answers"]
                level = float(a["priority"]["score"])  # expected level, 0 = low .. 2 = high
                out.append({"queue": a["route"]["choice"], "queue_conf": a["route"]["confidence"],
                            "queue_probs": a["route"]["probabilities"], "priority_level": level,
                            "priority": PRIORITIES[min(2, max(0, round(level)))],  # levels are 0-based
                            "priority_probs": a["priority"]["probabilities"], "escalate": a["escalate"]["noul"]})
        return out

    @torch.no_grad()
    def same_problem(self, new: tuple[str, str], past: list[tuple[str, str, str]], bs: int = 32) -> list[float]:
        """P(yes) that each past (subject, body, answer) ticket is the same problem as the new one."""
        reqs = [({"new_ticket": ticket_state(*new), "past_ticket": {**ticket_state(p[0], p[1]), "resolution": p[2]}},
                 {"same": SAME}) for p in past]
        out = []
        for s in range(0, len(reqs), bs):
            out += [r["answers"]["same"]["noul"] for r in self.model.system_one_batch(reqs[s:s + bs])]
        return out

    @torch.no_grad()
    def grounded(self, new: tuple[str, str], cases: list[dict], draft: str) -> float:
        state = {"new_ticket": ticket_state(*new), "past_tickets": cases, "draft_reply": draft}
        return self.model.system_one(state, {"grounded": GROUNDED})["answers"]["grounded"]["noul"]
