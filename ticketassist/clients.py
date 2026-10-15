"""Swappable embedder and reply writer: local Hugging Face models or any OpenAI-compatible API (config.py).

Both API clients speak the OpenAI wire format, which most providers and local servers accept:
  POST {base}/embeddings        {"model", "input": [...]}                     -> data[i].embedding
  POST {base}/chat/completions  {"model", "messages", "stream": true, ...}    -> server-sent events with delta.content
"""

import json

import httpx
import numpy as np

from .config import CONFIG, Config


# ---------------------------------------------------------------- embedders
class LocalEmbedder:
    """sentence-transformers model; EmbeddingGemma gets its retrieval prompts, other models plain text."""

    def __init__(self, cfg: Config):
        import torch
        from sentence_transformers import SentenceTransformer
        gemma = "embeddinggemma" in cfg.embedder.lower()
        kw = {"config_kwargs": {"vision_config": None, "audio_config": None}} if gemma else {}
        self.model = SentenceTransformer(cfg.embedder, device=cfg.device, model_kwargs={"torch_dtype": torch.bfloat16}, **kw)
        self.q_prompt = "task: search result | query: " if gemma else ""

    def documents(self, texts):
        return np.asarray(self.model.encode(list(texts), normalize_embeddings=True))

    def query(self, text):
        return np.asarray(self.model.encode([text], prompt=self.q_prompt or None, normalize_embeddings=True)[0])


class ApiEmbedder:
    """OpenAI-compatible /embeddings endpoint; vectors are L2-normalised so cosine = dot product."""

    def __init__(self, cfg: Config):
        if not cfg.embed_api_key and "localhost" not in cfg.embed_api_base and "127.0.0.1" not in cfg.embed_api_base:
            raise ValueError("TA_EMBEDDER=api needs TA_EMBED_API_KEY (or OPENAI_API_KEY)")
        self.cfg = cfg
        self.http = httpx.Client(base_url=cfg.embed_api_base.rstrip("/"), timeout=60,
                                 headers={"Authorization": f"Bearer {cfg.embed_api_key}"} if cfg.embed_api_key else {})

    def _embed(self, texts):
        out = []
        for i in range(0, len(texts), 64):
            r = self.http.post("/embeddings", json={"model": self.cfg.embed_api_model, "input": texts[i:i + 64]})
            r.raise_for_status()
            out += [d["embedding"] for d in sorted(r.json()["data"], key=lambda d: d["index"])]
        v = np.asarray(out, dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def documents(self, texts):
        return self._embed(list(texts))

    def query(self, text):
        return self._embed([text])[0]


def make_embedder(cfg: Config = CONFIG):
    return ApiEmbedder(cfg) if cfg.embedder == "api" else LocalEmbedder(cfg)


# ---------------------------------------------------------------- reply writers
class LocalWriter:
    """A Hugging Face chat model on its own GPU (default ALLaM-7B-Instruct, 4-bit NF4)."""

    def __init__(self, cfg: Config):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        kw = {"dtype": torch.bfloat16}
        if cfg.llm_4bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_compute_dtype=torch.bfloat16)
        self.cfg, self.device = cfg, cfg.llm_device
        self.tok = AutoTokenizer.from_pretrained(cfg.llm_model)
        self.model = AutoModelForCausalLM.from_pretrained(cfg.llm_model, device_map={"": cfg.llm_device}, **kw).eval()
        self.name = cfg.llm_model.split("/")[-1]

    def stream(self, messages):
        """Yields the reply written so far, every few tokens (greedy decoding)."""
        from threading import Thread

        import torch
        from transformers import TextIteratorStreamer
        ids = self.tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(self.device)
        streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)

        def gen():
            with torch.inference_mode():
                self.model.generate(**ids, max_new_tokens=self.cfg.llm_max_tokens, do_sample=False,
                                    repetition_penalty=1.05, streamer=streamer)
        th = Thread(target=gen, daemon=True)
        th.start()
        text = ""
        for piece in streamer:
            text += piece
            yield text.strip()
        th.join()


class ApiWriter:
    """Any OpenAI-compatible chat endpoint, streamed. temperature 0 for repeatable replies."""

    def __init__(self, cfg: Config):
        if not cfg.llm_api_key and "localhost" not in cfg.llm_api_base and "127.0.0.1" not in cfg.llm_api_base:
            raise ValueError("TA_LLM=api needs TA_LLM_API_KEY (or OPENAI_API_KEY)")
        self.cfg = cfg
        self.http = httpx.Client(base_url=cfg.llm_api_base.rstrip("/"), timeout=120,
                                 headers={"Authorization": f"Bearer {cfg.llm_api_key}"} if cfg.llm_api_key else {})
        self.name = cfg.llm_api_model

    def stream(self, messages):
        body = {"model": self.cfg.llm_api_model, "messages": messages, "stream": True, "temperature": 0,
                "max_tokens": self.cfg.llm_max_tokens}
        text = ""
        with self.http.stream("POST", "/chat/completions", json=body) as r:
            if r.status_code >= 400:
                r.read()
                raise RuntimeError(f"reply API error {r.status_code}: {r.text[:300]}")
            for line in r.iter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                choices = json.loads(data).get("choices") or [{}]
                piece = (choices[0].get("delta") or {}).get("content") or ""
                if piece:
                    text += piece
                    yield text.strip()


def make_writer(cfg: Config = CONFIG):
    return ApiWriter(cfg) if cfg.llm == "api" else LocalWriter(cfg)
