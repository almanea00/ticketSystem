"""Ticket priority for d1-3B, following the priority rules of arabic_support_natural_10000.csv (priority_reason)."""
PRIORITY_Q = {"type": "choice", "instructions": "How urgent is this customer-support ticket?", "criteria": {
    "high": ("High: the customer reports an active threat right now: their account was taken over or hacked, money is "
             "being taken or used without their permission (unauthorized payments), or a product is causing a physical "
             "safety hazard (fire, smoke, electric shock, injury, overheating)."),
    "medium": ("Medium: money or a delivery went wrong but nothing is dangerous: charged twice, charged without an order, "
               "a disputed subscription charge, a refund or funds still pending, an installment amount or schedule that "
               "is wrong, a missed promised delivery date, an order marked delivered but not received, a missing, wrong "
               "or damaged item, or a product that stopped working (no hazard)."),
    "low": ("Low: a routine request with no urgent harm: questions, tracking, changing an order or address, cancelling, "
            "returns or exchanges, invoices, coupons, loyalty points, account or password help, app errors, shipping or "
            "customs fees, product availability or usage, unsubscribing, privacy requests, general complaints."),
}}
