"""Every model, path and setting of the ticket assistant in one place.

Values come from environment variables (or a `.env` file in the project root); nothing needs editing in the code.
Copy `.env.example` to `.env` and change what you need. The three model slots:

  decision model  TA_DECISION = local | api              a d1-style model by link (TA_DECISION_MODEL: Hugging Face id
                                                         or folder; default our fine-tuned runs/d1_merged_all), or the
                                                         same model behind a vLLM server (TA_DECISION_API_BASE/_KEY/_MODEL)
  embedder        TA_EMBEDDER = <Hugging Face id> | api   local sentence-transformers model, or any OpenAI-compatible
                                                         /embeddings endpoint (TA_EMBED_API_BASE/_KEY/_MODEL)
  reply writer    TA_LLM = local | api                   local Hugging Face chat model (TA_LLM_MODEL), or any
                                                         OpenAI-compatible /chat/completions endpoint
                                                         (TA_LLM_API_BASE/_KEY/_MODEL): OpenAI, Azure OpenAI,
                                                         Groq, Together, OpenRouter, a vLLM or Ollama server, ...
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path = ROOT / ".env") -> None:
    """KEY=VALUE lines; real environment variables win over the file."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()


def _env(name, default):
    return os.environ.get(name, default)


@dataclass
class Config:
    # decision model: a d1-style model given by link (Hugging Face id or local folder), run here ("local"), or the
    # same model served by vLLM ("api": TA_DECISION_API_BASE / _KEY / _MODEL; the link still names its tokenizer)
    decision: str = field(default_factory=lambda: _env("TA_DECISION", "local"))
    decision_model: str = field(default_factory=lambda: _env("TA_DECISION_MODEL", "runs/d1_merged_all"))
    device: str = field(default_factory=lambda: _env("TA_DEVICE", "cuda:0"))
    decision_api_base: str = field(default_factory=lambda: _env("TA_DECISION_API_BASE", "http://127.0.0.1:18031/v1"))
    decision_api_key: str = field(default_factory=lambda: _env("TA_DECISION_API_KEY", ""))
    decision_api_model: str = field(default_factory=lambda: _env("TA_DECISION_API_MODEL", "d1-ticket"))
    questions: str = field(default_factory=lambda: _env("TA_QUESTIONS", "ticketassist/questions.json"))   # intent / team / priority

    # embedder: a Hugging Face sentence-transformers id, or "api"
    embedder: str = field(default_factory=lambda: _env("TA_EMBEDDER", "google/embeddinggemma-2"))
    embed_api_base: str = field(default_factory=lambda: _env("TA_EMBED_API_BASE", "https://api.openai.com/v1"))
    embed_api_key: str = field(default_factory=lambda: _env("TA_EMBED_API_KEY", _env("OPENAI_API_KEY", "")))
    embed_api_model: str = field(default_factory=lambda: _env("TA_EMBED_API_MODEL", "text-embedding-3-small"))

    # reply writer: "local" (Hugging Face chat model on a GPU) or "api" (OpenAI-compatible chat endpoint)
    llm: str = field(default_factory=lambda: _env("TA_LLM", "local"))
    llm_model: str = field(default_factory=lambda: _env("TA_LLM_MODEL", "ALLaM-AI/ALLaM-7B-Instruct-preview"))
    llm_device: str = field(default_factory=lambda: _env("TA_LLM_DEVICE", "cuda:1"))
    llm_4bit: bool = field(default_factory=lambda: _env("TA_LLM_4BIT", "1") == "1")
    llm_api_base: str = field(default_factory=lambda: _env("TA_LLM_API_BASE", "https://api.openai.com/v1"))
    llm_api_key: str = field(default_factory=lambda: _env("TA_LLM_API_KEY", _env("OPENAI_API_KEY", "")))
    llm_api_model: str = field(default_factory=lambda: _env("TA_LLM_API_MODEL", "gpt-4o-mini"))
    llm_max_tokens: int = field(default_factory=lambda: int(_env("TA_LLM_MAX_TOKENS", "220")))

    # knowledge base and decision settings
    kb_dir: str = field(default_factory=lambda: _env("TA_KB_DIR", "data/course/kb"))
    top_k: int = field(default_factory=lambda: int(_env("TA_TOP_K", "4")))
    guard_block: float = field(default_factory=lambda: float(_env("TA_GUARD_BLOCK", "0.2")))
    guard_support: float = field(default_factory=lambda: float(_env("TA_GUARD_SUPPORT", "0.05")))
    answer_min: float = field(default_factory=lambda: float(_env("TA_ANSWER_MIN", "0.3")))

    def path(self, p: str) -> Path:
        q = Path(p)
        return q if q.is_absolute() else ROOT / q

    def describe(self) -> str:
        llm = f"{self.llm_api_model} via {self.llm_api_base}" if self.llm == "api" else self.llm_model
        emb = f"{self.embed_api_model} via {self.embed_api_base}" if self.embedder == "api" else self.embedder
        dec = f"{self.decision_api_model} via {self.decision_api_base}" if self.decision == "api" else self.decision_model
        return f"decision: {dec} · embedder: {emb} · reply: {llm}"


CONFIG = Config()
