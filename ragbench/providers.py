"""
providers.py — откуда берутся эмбеддинги и генерация.

Две независимые сущности:
  • Embedder — текст -> вектор. Либо нейросетевой (через OpenAI-совместимый
    API: LM Studio, GitHub Models, прокси), либо offline TF-IDF на numpy.
  • LLM — генерация ответа. Либо API, либо NullLLM (offline, без сети).

Смысл разделения: retrieval и eval должны прогоняться даже без сервера и без
интернета. Поэтому offline-эмбеддер обучаемый (fit на корпусе), а нейросетевой —
stateless. Общий интерфейс: .fit(corpus) и .encode(texts) -> (N, d).
"""
from __future__ import annotations

import re
import math
import time
import numpy as np


def _with_retry(fn, tries: int = 4, base: float = 2.0):
    """Повтор с экспоненциальной паузой — переживаем rate-limit/сетевые сбои API."""
    for attempt in range(tries):
        try:
            return fn()
        except Exception as e:
            transient = any(s in str(e).lower() for s in
                            ("rate", "429", "timeout", "temporar", "503", "overload"))
            if attempt == tries - 1 or not transient:
                raise
            time.sleep(base * (2 ** attempt))


# --- OFFLINE TF-IDF ---

_WORD_RE = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)


class TfidfEmbedder:
    """
    Классический TF-IDF на чистом numpy — стенд-ин для нейро-эмбеддера, когда
    нет ни LM Studio, ни сети. Признаки двух типов:
      • слова: униграммы + биграммы (ловят терминологию «суточные», «отпуск»);
      • символьные n-граммы 3..5 (ловят «10:00», «1,5x», опечатки, падежи).
    Символьные n-граммы — то, что делает offline-поиск устойчивым к морфологии
    русского языка без стеммера.
    """

    name = "offline-tfidf"
    has_state = True

    def __init__(self, char_ngrams=(3, 4, 5), word_ngrams=(1, 2)):
        self.char_ngrams = tuple(char_ngrams)
        self.word_ngrams = tuple(word_ngrams)
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray | None = None

    # --- извлечение признаков из одного текста ---
    def _features(self, text: str) -> dict[str, int]:
        text = text.lower()
        words = _WORD_RE.findall(text)
        feats: dict[str, int] = {}

        for n in self.word_ngrams:
            for i in range(len(words) - n + 1):
                key = "w:" + "_".join(words[i:i + n])
                feats[key] = feats.get(key, 0) + 1

        # символьные n-граммы по «схлопнутому» тексту (пробелы как один _)
        squashed = "_".join(words)
        for n in self.char_ngrams:
            for i in range(len(squashed) - n + 1):
                key = "c:" + squashed[i:i + n]
                feats[key] = feats.get(key, 0) + 1
        return feats

    def fit(self, corpus: list[str]) -> "TfidfEmbedder":
        df: dict[str, int] = {}
        per_doc = [self._features(t) for t in corpus]
        for feats in per_doc:
            for key in feats:
                df[key] = df.get(key, 0) + 1
        self.vocab = {key: j for j, key in enumerate(sorted(df))}
        n = max(1, len(corpus))
        idf = np.zeros(len(self.vocab), dtype=np.float32)
        for key, j in self.vocab.items():
            # сглаженный idf: редкое слово важнее частого
            idf[j] = math.log((n + 1) / (df[key] + 1)) + 1.0
        self.idf = idf
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.idf is None:
            raise RuntimeError("TfidfEmbedder.encode вызван до fit()")
        out = np.zeros((len(texts), len(self.vocab)), dtype=np.float32)
        for row, text in enumerate(texts):
            for key, tf in self._features(text).items():
                j = self.vocab.get(key)
                if j is not None:
                    out[row, j] = tf * self.idf[j]
            norm = np.linalg.norm(out[row])
            if norm > 0:
                out[row] /= norm
        return out

    # сериализация состояния, чтобы переиспользовать индекс между запусками
    def state(self) -> dict:
        return {
            "kind": "tfidf",
            "char_ngrams": list(self.char_ngrams),
            "word_ngrams": list(self.word_ngrams),
            "vocab": self.vocab,
            "idf": self.idf.tolist() if self.idf is not None else None,
        }

    @classmethod
    def from_state(cls, d: dict) -> "TfidfEmbedder":
        emb = cls(char_ngrams=d["char_ngrams"], word_ngrams=d["word_ngrams"])
        emb.vocab = d["vocab"]
        emb.idf = np.asarray(d["idf"], dtype=np.float32) if d["idf"] else None
        return emb


# --- API-ЭМБЕДДЕР ---

class ApiEmbedder:
    """Нейросетевой эмбеддер через OpenAI-совместимый /v1/embeddings."""

    has_state = False

    def __init__(self, client, model: str):
        self.client = client
        self.model = model
        self.name = f"api:{model}"

    def fit(self, corpus: list[str]) -> "ApiEmbedder":
        return self  # stateless

    def encode(self, texts: list[str], batch: int = 64) -> np.ndarray:
        vecs: list[np.ndarray] = []
        for i in range(0, len(texts), batch):
            chunk = texts[i:i + batch]
            resp = _with_retry(lambda: self.client.embeddings.create(model=self.model, input=chunk))
            # порядок ответа гарантирован по data[i].index, но обычно совпадает
            data = sorted(resp.data, key=lambda d: d.index)
            for d in data:
                vecs.append(np.asarray(d.embedding, dtype=np.float32))
        return np.vstack(vecs)

    def state(self) -> dict:
        return {"kind": "api", "model": self.model}


# --- LLM ---

class ApiLLM:
    """Генерация через OpenAI-совместимый /v1/chat/completions."""

    has_llm = True

    def __init__(self, client, model: str):
        self.client = client
        self.model = model
        self.name = model

    def chat(self, messages, temperature=0.3, max_tokens=512) -> str:
        resp = _with_retry(lambda: self.client.chat.completions.create(
            model=self.model, messages=messages,
            temperature=temperature, max_tokens=max_tokens,
        ))
        return resp.choices[0].message.content or ""

    def stream(self, messages, temperature=0.3, max_tokens=512):
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages,
            temperature=temperature, max_tokens=max_tokens, stream=True,
        )
        for chunk in resp:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


class NullLLM:
    """Заглушка для offline-режима: генерации нет, ответ собирается экстрактивно."""

    has_llm = False
    name = "none (extractive)"

    def chat(self, *a, **k) -> str:
        raise RuntimeError("LLM недоступен в offline-режиме")

    def stream(self, *a, **k):
        raise RuntimeError("LLM недоступен в offline-режиме")
