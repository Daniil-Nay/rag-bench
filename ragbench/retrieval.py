"""
retrieval.py — стратегии поиска. Главная идея ДЗ: retrieval — это не одна
функция, а сменный модуль. Шесть режимов, у каждого свой характер:

  naive       — лекционный dense top-k (косинус). Базлайн для сравнения.
  bm25        — чисто лексический поиск. Силён на числах/именах/тегах.
  hybrid      — dense + bm25, слитые через Reciprocal Rank Fusion (RRF).
  hyde        — HyDE: LLM пишет гипотетический фрагмент-ответ, ищем по нему.
  multiquery  — LLM делает 3 перефразировки вопроса, ранги сливаются RRF.
  rerank      — берём пул кандидатов и переупорядочиваем: LLM-судья, а без
                LLM — MMR (максимальная маржинальная релевантность, диверсификация).

Без LLM (offline) hyde/multiquery/llm-rerank честно деградируют до hybrid/MMR —
и об этом пишется в notes, а не молча. Прозрачность деградации — часть задания.
"""
from __future__ import annotations

import re
import json
import numpy as np
from dataclasses import dataclass, field

from .ingest import Chunk
from .providers import NullLLM

RRF_K = 60
MODES = ["naive", "bm25", "hybrid", "hyde", "multiquery", "rerank"]


@dataclass
class Hit:
    id: int
    score: float
    chunk: Chunk


@dataclass
class RetrievalResult:
    mode: str
    hits: list[Hit]
    notes: list[str] = field(default_factory=list)
    expansions: list[str] = field(default_factory=list)  # HyDE-текст / под-запросы


# --- низкоуровневые ранги ---

def _order(scores: np.ndarray, n: int) -> list[int]:
    return [int(i) for i in np.argsort(scores)[::-1][:n]]


def _rrf(rank_lists: list[list[int]], k: int) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion: устойчиво сливает несколько ранжирований."""
    agg: dict[int, float] = {}
    for order in rank_lists:
        for rank, idx in enumerate(order):
            agg[idx] = agg.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)
    return sorted(agg.items(), key=lambda kv: -kv[1])[:k]


def _hits(index, pairs: list[tuple[int, float]]) -> list[Hit]:
    return [Hit(i, float(s), index.chunks[i]) for i, s in pairs]


# --- базовые стратегии ---

def dense(index, query: str, k: int) -> list[Hit]:
    s = index.dense_scores(index.embed_query(query))
    return [Hit(i, float(s[i]), index.chunks[i]) for i in _order(s, k)]


def bm25(index, query: str, k: int) -> list[Hit]:
    s = index.bm25_scores(query)
    return [Hit(i, float(s[i]), index.chunks[i]) for i in _order(s, k)]


def hybrid(index, query: str, k: int, pool: int) -> list[Hit]:
    ds = index.dense_scores(index.embed_query(query))
    bs = index.bm25_scores(query)
    fused = _rrf([_order(ds, pool), _order(bs, pool)], k)
    return _hits(index, fused)


# --- query-трансформации ---

def _hyde(index, query, k, pool, llm, notes):
    if not llm.has_llm:
        notes.append("HyDE отключён (нет LLM) -> hybrid")
        return hybrid(index, query, k, pool), []
    prompt = [{"role": "user", "content":
               "Напиши правдоподобный короткий фрагмент внутреннего регламента "
               "компании (2-3 предложения), который мог бы содержать ответ на вопрос. "
               "Не рассуждай, выдай только фрагмент.\n\nВопрос: " + query}]
    hypo = llm.chat(prompt, temperature=0.3, max_tokens=200).strip()
    ds = index.dense_scores(index.embed_query(query + "\n" + hypo))
    bs = index.bm25_scores(query)
    fused = _rrf([_order(ds, pool), _order(bs, pool)], k)
    return _hits(index, fused), [hypo]


def _multiquery(index, query, k, pool, llm, notes):
    if not llm.has_llm:
        notes.append("Multiquery отключён (нет LLM) -> hybrid")
        return hybrid(index, query, k, pool), []
    prompt = [{"role": "user", "content":
               "Перефразируй вопрос 3 разными способами для поиска по документам. "
               "Один вариант на строку, без нумерации и пояснений.\n\nВопрос: " + query}]
    raw = llm.chat(prompt, temperature=0.5, max_tokens=160)
    subs = [ln.strip(" -•\t") for ln in raw.splitlines() if ln.strip()][:3]
    variants = [query] + subs
    ranks = []
    for v in variants:
        ranks.append(_order(index.dense_scores(index.embed_query(v)), pool))
        ranks.append(_order(index.bm25_scores(v), pool))
    return _hits(index, _rrf(ranks, k)), subs


# --- rerank ---

def _mmr(index, query, candidates: list[Hit], k: int, lam: float = 0.7) -> list[Hit]:
    """MMR: релевантность минус избыточность. Работает без LLM, по dense-векторам."""
    qn = index.embed_query(query)
    qn = qn / (np.linalg.norm(qn) or 1.0)
    cand = [h.id for h in candidates]
    docv = index._normed[cand]
    rel = docv @ qn
    chosen: list[int] = []
    chosen_local: list[int] = []
    while len(chosen) < min(k, len(cand)):
        best_j, best_val = -1, -1e9
        for j in range(len(cand)):
            if j in chosen_local:
                continue
            div = max((float(docv[j] @ docv[c]) for c in chosen_local), default=0.0)
            val = lam * float(rel[j]) - (1 - lam) * div
            if val > best_val:
                best_val, best_j = val, j
        chosen_local.append(best_j)
        chosen.append(cand[best_j])
    return [Hit(i, float(rel[cand.index(i)]), index.chunks[i]) for i in chosen]


def _llm_rerank(index, query, candidates: list[Hit], k: int, llm, notes) -> list[Hit]:
    listing = "\n".join(f"[#{h.id}] {h.chunk.text[:280]}" for h in candidates)
    prompt = [{"role": "user", "content":
               f"Оцени релевантность каждого фрагмента вопросу по шкале 0-10. "
               f"Верни ТОЛЬКО JSON-массив [{{\"id\":int,\"score\":int}}].\n\n"
               f"Вопрос: {query}\n\nФрагменты:\n{listing}"}]
    try:
        raw = llm.chat(prompt, temperature=0.0, max_tokens=300)
        m = re.search(r"\[.*\]", raw, re.S)
        scores = {int(d["id"]): float(d["score"]) for d in json.loads(m.group(0))}
        ranked = sorted(candidates, key=lambda h: -scores.get(h.id, -1))
        return ranked[:k]
    except Exception as e:
        notes.append(f"LLM-rerank не распарсился ({e.__class__.__name__}) -> MMR")
        return _mmr(index, query, candidates, k)


def _rerank(index, query, k, pool, llm, notes):
    candidates = hybrid(index, query, pool, pool)
    if llm.has_llm:
        return _llm_rerank(index, query, candidates, k, llm, notes)
    notes.append("LLM-rerank недоступен (нет LLM) -> MMR")
    return _mmr(index, query, candidates, k)


# --- публичный диспетчер ---

def retrieve(index, query: str, mode: str = "hybrid", k: int = 3,
             llm=None, pool: int = 12) -> RetrievalResult:
    llm = llm or NullLLM()
    notes: list[str] = []
    expansions: list[str] = []

    if mode in ("naive", "dense"):
        hits = dense(index, query, k)
    elif mode == "bm25":
        hits = bm25(index, query, k)
    elif mode == "hybrid":
        hits = hybrid(index, query, k, pool)
    elif mode == "hyde":
        hits, expansions = _hyde(index, query, k, pool, llm, notes)
    elif mode == "multiquery":
        hits, expansions = _multiquery(index, query, k, pool, llm, notes)
    elif mode == "rerank":
        hits = _rerank(index, query, k, pool, llm, notes)
    else:
        raise ValueError(f"Неизвестный режим retrieval: {mode!r}. Доступно: {MODES}")

    return RetrievalResult(mode=mode, hits=hits, notes=notes, expansions=expansions)
