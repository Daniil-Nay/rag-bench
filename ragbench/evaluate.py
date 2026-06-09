"""
evaluate.py — измеримое качество RAG, а не «на глаз».

Зачем: лекционный скрипт нельзя сравнить сам с собой — непонятно, помогает
ли hybrid и не ломает ли rerank. Здесь есть gold-набор (data/eval_set.json) и
метрики, которые считаются на реальном прогоне:

  Retrieval (по needle — дословному фрагменту релевантного раздела):
    Hit@1  — релевантный раздел стоит первым
    Hit@3  — релевантный раздел попал в топ-k
    MRR    — 1 / ранг первого релевантного (награда за высокий ранг)

  End-to-end:
    Ans    — итоговый ответ содержит нужный факт (а для вопроса-ловушки без
             ответа в документах — корректно сработал guardrail)

run_bench прогоняет все режимы и печатает их рядом — видно цену каждой идеи.
"""
from __future__ import annotations

import os
import json

from .retrieval import retrieve, MODES
from .answer import answer


def load_eval_set(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)["questions"]


def _relevant(text: str, needles: list[str]) -> bool:
    return any(n in text for n in needles)


def _fact_ok(ans_text: str, fact: str) -> bool:
    a = ans_text.lower()
    if fact.strip().lower() == "нет":          # вопрос-ловушка
        return "ответа нет" in a
    return fact.lower() in a


def evaluate_mode(index, questions: list[dict], mode: str, llm, cfg) -> dict:
    k = cfg.top_k
    h1 = h3 = mrr = retr_n = 0
    ans_ok = 0
    for q in questions:
        rr = retrieve(index, q["q"], mode=mode, k=k, llm=llm, pool=cfg.candidate_pool)

        # retrieval-метрики только для вопросов с размеченным needle
        if q["needles"]:
            rank = 0
            for i, h in enumerate(rr.hits):
                if _relevant(h.chunk.text, q["needles"]):
                    rank = i + 1
                    break
            h1 += int(rank == 1)
            h3 += int(0 < rank <= k)
            mrr += (1.0 / rank) if rank else 0.0
            retr_n += 1

        # end-to-end ответ (для всех вопросов, включая ловушку)
        a = answer(index, q["q"], rr, llm=llm, cfg=cfg)
        ans_ok += int(_fact_ok(a.text, q["fact"]))

    retr_n = max(1, retr_n)
    return {
        "mode": mode,
        "hit1": h1 / retr_n,
        "hit3": h3 / retr_n,
        "mrr": mrr / retr_n,
        "ans": ans_ok / len(questions),
    }


def no_rag_baseline(questions: list[dict], llm, cfg) -> dict | None:
    """Контроль: LLM отвечает БЕЗ контекста. Показывает, что retrieval реально нужен."""
    if not getattr(llm, "has_llm", False):
        return None
    ok = 0
    for q in questions:
        text = llm.chat(
            [{"role": "user", "content":
              "Ответь кратко. Если не знаешь точно — скажи «ответа нет».\n\n" + q["q"]}],
            temperature=0.0, max_tokens=120)
        ok += int(_fact_ok(text, q["fact"]))
    return {"mode": "no-rag (LLM only)", "hit1": float("nan"), "hit3": float("nan"),
            "mrr": float("nan"), "ans": ok / len(questions)}


def run_bench(index, questions: list[dict], llm, cfg, modes=None) -> list[dict]:
    modes = modes or MODES
    rows = [evaluate_mode(index, questions, m, llm, cfg) for m in modes]
    base = no_rag_baseline(questions, llm, cfg)
    if base:
        rows.append(base)
    return rows


def render_table(rows: list[dict]) -> str:
    head = f"{'режим':<18} {'Hit@1':>7} {'Hit@3':>7} {'MRR':>7} {'Ans':>7}"
    lines = [head, "-" * len(head)]
    for r in rows:
        def cell(x):
            return "  —  " if x != x else f"{x:.3f}"   # x!=x ловит NaN
        lines.append(f"{r['mode']:<18} {cell(r['hit1']):>7} {cell(r['hit3']):>7} "
                     f"{cell(r['mrr']):>7} {cell(r['ans']):>7}")
    return "\n".join(lines)


def save_results(rows: list[dict], info: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"provider": info, "rows": rows}, f, ensure_ascii=False, indent=2)
