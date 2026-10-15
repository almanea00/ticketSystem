# Ticket Assistant

Screens, classifies and answers customer-support tickets for an online store, in Arabic (فصحى and Gulf dialects) and
English. Built for the [SDAIA Academy](https://github.com/SDAIAAcademy) Applied NLP capstone (Ticket Assistant use case,
[SDAIAAcademy/applied-nlp](https://github.com/SDAIAAcademy/applied-nlp)).

![Pipeline](docs/figures/pipeline.png)

1. **Safeguard** (decision model): blocks prompt injections and harmful requests, and answers off-topic messages with
   what the store can help with.
2. **Decisions** (decision model): intent (40), team (13) and priority (low / medium / high). The app uses them to
   order the queue and pick the team inbox, and the reply writer gets them in its prompt.
3. **Retrieval** (EmbeddingGemma): the reply writer first rewrites the message as a clear search query (few-shot),
   then the 4 closest help-center sections are found.
4. **Reply writer** (ALLaM-7B or any OpenAI-compatible API): answers from those sections only and cites them as [n].

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
- **The decision model** (safeguard, intent, team, priority), one of:
  - **our fine-tuned d1-3B on an NVIDIA GPU** (about 10 GB with the embedder), in the app or on your own vLLM server
    (`TA_DECISION=local` or `api`). This is the model the report's numbers are for.
  - **a hosted decision API in the same System One format, such as TypeSafe's Jev**, with no GPU for decisions
    (zero-shot, so the report's decision numbers do not apply to it):

```bash
TA_DECISION=systemone
TA_DECISION_API_BASE=https://api.typesafe.ai/v1
TA_DECISION_API_KEY=...
TA_DECISION_API_MODEL=jev-latest
```

  A general chat API such as OpenAI's chat completions does not work here: it does not return a probability for every
  option.
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
