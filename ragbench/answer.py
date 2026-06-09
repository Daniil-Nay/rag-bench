"""
answer.py — сборка ответа поверх найденных чанков.

Добавлено к лекции:
  • цитаты [#id] прямо в ответе — видно, откуда взят факт;
  • оценка уверенности по косинусу лучшего чанка + guardrail: если уверенность
    ниже порога, честно отвечаем «в документах нет» и показываем ближайшее;
  • экстрактивный режим без LLM — выбираем предложения из топ-чанков по
    пересечению с вопросом, чтобы offline-демо тоже давало осмысленный ответ;
  • стриминг (нужен веб-интерфейсу с SSE).
"""
from __future__ import annotations

import re
import numpy as np
from dataclasses import dataclass, field

from .retrieval import RetrievalResult, Hit
from .providers import NullLLM

_TOK = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)
_SENT = re.compile(r"(?<=[.!?])\s+")              # НЕ режем по «:» (адрес отрывается от «адрес:»)
_ABBR_END = re.compile(r"(?:^|\s)[A-Za-zА-Яа-яЁё]{1,2}\.$")  # отдельное «г.»/«ул.»/«д.» — не конец предложения
_HEADING_LINE = re.compile(r"^\s*(?:\d+\.\s+)?[А-ЯЁA-Z«][А-ЯЁA-Z0-9 ,«»\-/]{3,}$")
_GUARD = "В предоставленных документах ответа нет."


def _split_sentences(text: str) -> list[str]:
    """Деление на предложения с учётом русских сокращений (склеиваем «г.» с адресом)."""
    out: list[str] = []
    for p in _SENT.split(text):
        if out and _ABBR_END.search(out[-1]):
            out[-1] += " " + p
        else:
            out.append(p)
    return out

PROMPT = """Ты отвечаешь на вопросы СТРОГО по контексту из внутренних документов компании.
Если ответа в контексте нет — ответь дословно: «{guard}»
Не выдумывай факты и не используй внешние знания. После ответа укажи источники в формате [#id].

КОНТЕКСТ:
{context}

ВОПРОС: {question}
ОТВЕТ:"""


@dataclass
class Answer:
    text: str
    hits: list[Hit]
    confidence: float
    conf_label: str
    grounded: bool                 # True = ответ LLM, False = экстрактивный
    notes: list[str] = field(default_factory=list)


def build_context(hits: list[Hit]) -> str:
    parts = []
    for h in hits:
        tag = f"[#{h.id}" + (f" · {h.chunk.section}" if h.chunk.section else "") + "]"
        parts.append(f"{tag}\n{h.chunk.text}")
    return "\n\n---\n\n".join(parts)


def confidence(index, query: str, hits: list[Hit]) -> tuple[float, str]:
    """Уверенность = косинус вопроса с лучшим найденным чанком (единая шкала для всех режимов)."""
    if not hits:
        return 0.0, "нет"
    ds = index.dense_scores(index.embed_query(query))
    top = max(float(ds[h.id]) for h in hits)
    top = max(0.0, min(1.0, top))
    label = "высокая" if top >= 0.45 else "средняя" if top >= 0.25 else "низкая"
    return top, label


def _extractive(index, query: str, hits: list[Hit], max_sents: int = 2) -> str:
    """Без LLM: вытащить из топ-чанков предложения, ближайшие к вопросу.

    Близость считаем тем же эмбеддером (TF-IDF с символьными n-граммами), а не
    пересечением слов — иначе морфология русского ломает матч («город» ≠ «города»,
    «российский» ≠ «России»). Предложения из топ-чанка (rank 0) получают бонус —
    ответ обычно живёт в лучшем по retrieval разделе, а не в соседнем.
    """
    qv = index.embed_query(query)
    qn = qv / (np.linalg.norm(qv) or 1.0)
    cands, seen = [], set()
    for h in hits:
        body = "\n".join(l for l in h.chunk.text.splitlines()
                         if not _HEADING_LINE.match(l.strip()))
        for s in _split_sentences(body):
            s = s.strip().lstrip("-—•").strip()
            if len(s) < 8 or s in seen:
                continue
            seen.add(s)
            sv = index.embed_query(s)
            sim = float(sv @ qn) / (np.linalg.norm(sv) or 1.0)
            cands.append((sim, h.id, s))
    if not cands:
        top = hits[0]
        return f"{top.chunk.text.split('.')[0].strip()}. [#{top.id}]"
    cands.sort(key=lambda x: -x[0])
    return " ".join(f"{s} [#{cid}]" for _, cid, s in cands[:max_sents])


def answer(index, query: str, rr: RetrievalResult, llm=None, cfg=None) -> Answer:
    llm = llm or NullLLM()
    conf, label = confidence(index, query, rr.hits)
    floor = getattr(cfg, "min_similarity", 0.12)

    if conf < floor or not rr.hits:
        closest = f" Ближе всего: [#{rr.hits[0].id}]." if rr.hits else ""
        return Answer(_GUARD + closest, rr.hits, conf, label, grounded=bool(llm.has_llm),
                      notes=rr.notes + ["сработал guardrail по низкой уверенности"])

    if llm.has_llm:
        prompt = PROMPT.format(guard=_GUARD, context=build_context(rr.hits), question=query)
        text = llm.chat([{"role": "user", "content": prompt}],
                        temperature=getattr(cfg, "temperature", 0.3),
                        max_tokens=getattr(cfg, "max_tokens", 512)).strip()
        return Answer(text or "[пустой ответ]", rr.hits, conf, label, grounded=True, notes=rr.notes)

    return Answer(_extractive(index, query, rr.hits), rr.hits, conf, label, grounded=False,
                  notes=rr.notes + ["ответ собран экстрактивно (offline, без LLM)"])


def stream_answer(index, query: str, rr: RetrievalResult, llm=None, cfg=None):
    """Генератор токенов для веб-интерфейса (SSE)."""
    llm = llm or NullLLM()
    conf, _ = confidence(index, query, rr.hits)
    floor = getattr(cfg, "min_similarity", 0.12)

    if conf < floor or not rr.hits:
        yield _GUARD
        return
    if llm.has_llm:
        prompt = PROMPT.format(guard=_GUARD, context=build_context(rr.hits), question=query)
        yield from llm.stream([{"role": "user", "content": prompt}],
                              temperature=getattr(cfg, "temperature", 0.3),
                              max_tokens=getattr(cfg, "max_tokens", 512))
    else:
        for word in _extractive(index, query, rr.hits).split(" "):
            yield word + " "
