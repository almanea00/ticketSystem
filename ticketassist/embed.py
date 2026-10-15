"""EmbeddingGemma-2 (text only) for tickets: one L2-normalized 768-dim vector per ticket.

Two ways to embed, compared in scripts/eval_retrieve.py:
  "search"      new ticket = query ("task: search result | query: "), past tickets = documents ("title: <subject> | text: <body>")
  "similarity"  both sides "task: sentence similarity | query: <subject>. <body>" (symmetric: ticket vs ticket)
"""

from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

MODEL = "google/embeddinggemma-2"
PROMPTS = {"search": ("task: search result | query: ", None), "similarity": ("task: sentence similarity | query: ",) * 2}


def ticket_text(subject: str, body: str) -> str:
    subject = (subject or "").strip()
    return f"{subject}. {body.strip()}" if subject else body.strip()


class Embedder:
    def __init__(self, device: str = "cuda", mode: str = "search", adapter: str | None = None):
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.model = SentenceTransformer(MODEL, device=device, model_kwargs={"torch_dtype": dtype},
                                         config_kwargs={"vision_config": None, "audio_config": None})
        self.model.max_seq_length = 512
        if adapter:
            from peft import PeftModel
            self.model[0].model = PeftModel.from_pretrained(self.model[0].model, Path(adapter), torch_device=device).merge_and_unload()
        self.mode = mode

    def _enc(self, texts, prompt, bs=64):
        return self.model.encode(texts, prompt=prompt, batch_size=bs, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=len(texts) > 2000).astype(np.float32)

    def queries(self, subjects, bodies) -> np.ndarray:
        return self._enc([ticket_text(s, b) for s, b in zip(subjects, bodies)], PROMPTS[self.mode][0])

    def documents(self, subjects, bodies) -> np.ndarray:
        if self.mode == "search":
            texts = [f"title: {(s or 'none').strip()} | text: {b.strip()}" for s, b in zip(subjects, bodies)]
            return self._enc(texts, None)
        return self._enc([ticket_text(s, b) for s, b in zip(subjects, bodies)], PROMPTS[self.mode][1])
