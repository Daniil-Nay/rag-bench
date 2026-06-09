"""
index.py — «индекс» поверх чанков: плотные векторы + лексический BM25.

В лекции индекс — это одна numpy-матрица и косинус. Здесь рядом живёт второй,
лексический индекс (BM25), потому что у dense и BM25 разные слепые зоны:
  • dense ловит смысл/перефраз («работа из дома» ≈ «удалённый формат»);
  • BM25 ловит точные токены и числа («2500», «@svp_parking»), где dense плывёт.
Гибрид (см. retrieval.py) объединяет оба ранга и поэтому стабильнее каждого.

Индекс умеет сохраняться на диск (.npz + .meta.json) вместе с состоянием
TF-IDF-эмбеддера, чтобы не пересчитывать всё на каждый запуск.
"""
from __future__ import annotations

import os
import re
import json
import math
import numpy as np

from .ingest import Chunk
from .providers import TfidfEmbedder

_TOK = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)
_K1, _B = 1.5, 0.75


def _normalize_rows(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


class Index:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, embedder):
        self.chunks = chunks
        self.vectors = vectors.astype(np.float32)
        self.embedder = embedder
        self._normed = _normalize_rows(self.vectors)
        self._prepare_bm25()

    # --- построение ---
    @classmethod
    def build(cls, chunks: list[Chunk], embedder) -> "Index":
        texts = [c.text for c in chunks]
        embedder.fit(texts)
        vectors = embedder.encode(texts)
        return cls(chunks, vectors, embedder)

    def _prepare_bm25(self) -> None:
        self._docs_tokens = [_TOK.findall(c.text.lower()) for c in self.chunks]
        self._doc_len = np.array([len(t) for t in self._docs_tokens], dtype=np.float32)
        self._avgdl = float(self._doc_len.mean()) if len(self._doc_len) else 0.0
        df: dict[str, int] = {}
        self._tf: list[dict[str, int]] = []
        for toks in self._docs_tokens:
            tf: dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            self._tf.append(tf)
            for term in tf:
                df[term] = df.get(term, 0) + 1
        n = max(1, len(self.chunks))
        self._idf = {t: math.log((n - d + 0.5) / (d + 0.5) + 1.0) for t, d in df.items()}

    # --- поиск ---
    def embed_query(self, query: str) -> np.ndarray:
        return self.embedder.encode([query])[0]

    def dense_scores(self, q_vec: np.ndarray) -> np.ndarray:
        qn = q_vec / (np.linalg.norm(q_vec) or 1.0)
        return self._normed @ qn

    def bm25_scores(self, query: str) -> np.ndarray:
        q_terms = _TOK.findall(query.lower())
        scores = np.zeros(len(self.chunks), dtype=np.float32)
        for i in range(len(self.chunks)):
            dl = self._doc_len[i]
            denom_norm = _K1 * (1 - _B + _B * dl / (self._avgdl or 1.0))
            s = 0.0
            tf = self._tf[i]
            for term in q_terms:
                f = tf.get(term)
                if not f:
                    continue
                s += self._idf.get(term, 0.0) * (f * (_K1 + 1)) / (f + denom_norm)
            scores[i] = s
        return scores

    # --- персистентность ---
    def save(self, path: str, extra: dict | None = None) -> None:
        np.savez_compressed(path, vectors=self.vectors)
        meta = {
            "dim": int(self.vectors.shape[1]),
            "chunks": [vars(c) for c in self.chunks],
            "embedder": self.embedder.state() if getattr(self.embedder, "has_state", False)
                        else {"kind": "api", "name": getattr(self.embedder, "name", "api")},
        }
        if extra:
            meta.update(extra)
        with open(path + ".meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)

    @staticmethod
    def read_meta(path: str) -> dict:
        with open(path + ".meta.json", encoding="utf-8") as f:
            return json.load(f)

    @classmethod
    def load(cls, path: str, embedder=None) -> "Index":
        with np.load(path) as npz:
            vectors = npz["vectors"]
        with open(path + ".meta.json", encoding="utf-8") as f:
            meta = json.load(f)
        chunks = [Chunk(**c) for c in meta["chunks"]]

        est = meta["embedder"]
        if est.get("kind") == "tfidf":
            embedder = TfidfEmbedder.from_state(est)  # восстанавливаем тот же словарь
        else:
            if embedder is None:
                raise RuntimeError("Для API-индекса нужен живой embedder при load()")
            if vectors.shape[1] != meta["dim"]:
                raise RuntimeError("Размерность эмбеддера изменилась — пересоберите индекс")
        return cls(chunks, vectors, embedder)

    @staticmethod
    def exists(path: str) -> bool:
        return os.path.exists(path) and os.path.exists(path + ".meta.json")
