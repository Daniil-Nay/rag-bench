"""
config.py — вся конфигурация и выбор провайдера в одном месте.

Провайдер выбирается переменной окружения RAG_PROVIDER:
    auto      (по умолчанию) — LM Studio если поднят на :1234, иначе offline
    lmstudio  — локальный LM Studio (http://127.0.0.1:1234/v1)
    github    — GitHub Models, ключ из GITHUB_TOKEN или `gh auth token`
    openai    — любой OpenAI-совместимый прокси (RAG_BASE_URL + RAG_API_KEY)
    offline   — без сети: TF-IDF эмбеддер + экстрактивный ответ без LLM

ВАЖНО: ключи берутся ТОЛЬКО из окружения / `gh auth token`. В коде и в
сохранённом индексе ключей нет — это сознательное требование безопасности.
"""
from __future__ import annotations

import os
import socket
import subprocess
from dataclasses import dataclass, field

from .providers import TfidfEmbedder, ApiEmbedder, ApiLLM, NullLLM

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(HERE, "data")


@dataclass
class Config:
    # данные
    data_dir: str = DATA_DIR
    doc_glob: str = "*.txt,*.md"
    index_path: str = os.path.join(DATA_DIR, "index.npz")
    # чанкинг
    chunk_size: int = 500
    chunk_overlap: int = 100
    # поиск / генерация
    top_k: int = 3
    candidate_pool: int = 12          # сколько кандидатов до reranker
    temperature: float = 0.3
    max_tokens: int = 512
    # порог уверенности: ниже — отвечаем «в документах нет»
    min_similarity: float = 0.15
    # провайдер
    provider: str = field(default_factory=lambda: os.environ.get("RAG_PROVIDER", "auto"))


def _port_open(host: str, port: int, timeout: float = 0.4) -> bool:
    s = socket.socket()
    s.settimeout(timeout)
    try:
        s.connect((host, port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def _gh_token() -> str | None:
    tok = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if tok:
        return tok
    try:  # достаём из gh CLI, не сохраняя никуда
        out = subprocess.run(["gh", "auth", "token"], capture_output=True,
                             text=True, timeout=5)
        tok = out.stdout.strip()
        return tok or None
    except Exception:
        return None


def resolve_provider(cfg: Config) -> str:
    p = (cfg.provider or "auto").lower()
    if p == "auto":
        return "lmstudio" if _port_open("127.0.0.1", 1234) else "offline"
    return p


def make_backends(cfg: Config):
    """
    Возвращает (embedder, llm, info_dict).
    Если выбранный провайдер недоступен — честно падаем в offline, чтобы демо
    всегда запускалось (важно для критерия «воспроизводимость»).
    """
    from openai import OpenAI
    provider = resolve_provider(cfg)
    info = {"requested": cfg.provider, "resolved": provider}

    def offline():
        info["resolved"] = "offline"
        info["embedder"] = "offline-tfidf"
        info["llm"] = "none (extractive)"
        return TfidfEmbedder(), NullLLM(), info

    if provider == "offline":
        return offline()

    if provider == "lmstudio":
        base = os.environ.get("RAG_BASE_URL", "http://127.0.0.1:1234/v1")
        emb_model = os.environ.get("RAG_EMBED_MODEL", "text-embedding-nomic-embed-text-v1.5")
        chat_model = os.environ.get("RAG_CHAT_MODEL", "google/gemma-4-e4b")
        client = OpenAI(base_url=base, api_key="lm-studio")

    elif provider == "github":
        tok = _gh_token()
        if not tok:
            print("[config] нет токена для GitHub Models -> offline")
            return offline()
        base = os.environ.get("RAG_BASE_URL", "https://models.github.ai/inference")
        emb_model = os.environ.get("RAG_EMBED_MODEL", "openai/text-embedding-3-small")
        chat_model = os.environ.get("RAG_CHAT_MODEL", "openai/gpt-4o-mini")
        client = OpenAI(base_url=base, api_key=tok)

    elif provider == "openai":
        base = os.environ.get("RAG_BASE_URL", "https://api.openai.com/v1")
        key = os.environ.get("RAG_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not key:
            print("[config] нет RAG_API_KEY/OPENAI_API_KEY -> offline")
            return offline()
        emb_model = os.environ.get("RAG_EMBED_MODEL", "text-embedding-3-small")
        chat_model = os.environ.get("RAG_CHAT_MODEL", "gpt-4o-mini")
        client = OpenAI(base_url=base, api_key=key)

    else:
        print(f"[config] неизвестный провайдер '{provider}' -> offline")
        return offline()

    info.update({"base_url": base, "embedder": f"api:{emb_model}", "llm": chat_model})
    return ApiEmbedder(client, emb_model), ApiLLM(client, chat_model), info
