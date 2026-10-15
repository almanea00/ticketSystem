# Ticket Assistant

Screens, classifies and answers customer-support tickets for an online store, in Arabic (فصحى and Gulf dialects) and
English. Built for the [SDAIA Academy](https://github.com/SDAIAAcademy) Applied NLP capstone (Ticket Assistant use case,
[SDAIAAcademy/applied-nlp](https://github.com/SDAIAAcademy/applied-nlp)).

```
message → ① safeguard → ② decisions (intent, team, priority) + ③ retrieval (query rewrite + EmbeddingGemma, top 4)
        → ④ reply writer (answers from the retrieved help-center sections, cited) → reply
```

- **Decision model:** [LiquidAI/d1-3B](https://huggingface.co/LiquidAI/d1-3B) fine-tuned with a LoRA (`runs/d1_lora_all`),
  merged into the weights with `python scripts/merge_lora.py` → `runs/d1_merged_all`.
- **Retriever:** google/embeddinggemma-2 over the course help center (`data/course/kb`).
- **Reply writer:** ALLaM-7B-Instruct, or any OpenAI-compatible API.

## Screenshots

**A prompt injection in Arabic is blocked by the safeguard** (the AI never sees it):

![Arabic prompt injection blocked](docs/screenshots/injection_blocked.png)

**A short dialect question, «وين فلوسي» ("where is my money?")**: the query is rewritten for the search, the refund
pages are found and the reply explains the timelines, citing them:

![Vague money question answered from the refund pages](docs/screenshots/vague_money_question.png)

**A return question, with every stage shown and the reply streaming in:**

![Return question streaming](docs/screenshots/return_streaming.png)

## Requirements

- **Python 3.12** and [uv](https://docs.astral.sh/uv/); internet on the first run to download the models.
- **An NVIDIA GPU with about 10 GB** for the decision model (d1-3B) and the embedder. The decision model cannot be
  replaced by an API key: it reads the probability of each option, which chat APIs do not return. It can run in
  the app or behind your own vLLM server (see `.env.example`).
- **The reply writer:** either a second GPU with about 6 GB (ALLaM-7B in 4-bit, the default `TA_LLM=local`), or an
  API key for any OpenAI-compatible provider. With an API key, one GPU is enough:

```bash
TA_LLM=api
TA_LLM_API_BASE=https://api.openai.com/v1     # or Groq, OpenRouter, Azure, Together, a local vLLM or Ollama server
TA_LLM_API_KEY=sk-...
TA_LLM_API_MODEL=gpt-4o-mini
```

On a single GPU with the local reply writer, set `TA_LLM_DEVICE=cuda:0` (it needs about 16 GB in total).

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
