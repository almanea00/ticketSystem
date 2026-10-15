# Ticket Assistant

Screens, classifies and answers customer-support tickets for an online store, in Arabic (فصحى and Gulf dialects) and
English. Built for the SDAIA Academy Applied NLP capstone (Ticket Assistant use case).

```
message → ① safeguard → ② decisions (intent, team, priority) + ③ retrieval (query rewrite + EmbeddingGemma, top 4)
        → ④ reply writer (answers from the retrieved help-center sections, cited) → reply
```

- **Decision model:** [LiquidAI/d1-3B](https://huggingface.co/LiquidAI/d1-3B) fine-tuned with a LoRA (`runs/d1_lora_all`),
  merged into the weights with `python scripts/merge_lora.py` → `runs/d1_merged_all`.
- **Retriever:** google/embeddinggemma-2 over the course help center (`data/course/kb`).
- **Reply writer:** ALLaM-7B-Instruct, or any OpenAI-compatible API.

## Run

```bash
uv sync
python scripts/merge_lora.py          # builds runs/d1_merged_all from the LoRA (once)
cp .env.example .env                  # every setting: models (a link or an API URL + key), help center, cut-offs
uv run python -m ticketassist.web     # http://127.0.0.1:7861  (One ticket and Queue tabs)
```

Python: `from ticketassist.system import TicketSystem; TicketSystem().handle("How long does a refund to my Visa card take?")`

Full documentation, data, baselines, settings and metrics: **[MODEL_CARD.md](MODEL_CARD.md)**.

## License

Code: Apache 2.0 (`LICENSE`). The models and datasets keep their own terms; see `NOTICE` and the model card
(d1-3B and the LoRA: LFM Open License v1.0).
