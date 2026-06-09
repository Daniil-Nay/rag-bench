#!/usr/bin/env python
"""
app.py — единый CLI для RAG-стенда.

    python app.py info                      какой провайдер/эмбеддер/LLM активен
    python app.py build [--chunk 500 --overlap 100]   собрать и сохранить индекс
    python app.py ask "вопрос" [--mode hybrid --k 3]  один ответ
    python app.py repl [--mode hybrid]                интерактивный режим (как в лекции)
    python app.py eval [--mode hybrid]                метрики одного режима
    python app.py bench                               сравнить ВСЕ режимы таблицей
    python app.py web [--port 8000]                   веб-интерфейс с источниками

Провайдер задаётся переменной RAG_PROVIDER (auto|lmstudio|github|openai|offline).
По умолчанию auto: LM Studio если запущен, иначе полностью offline.
"""
from __future__ import annotations

import sys
import os
import argparse

# Windows-консоль бывает в cp1251 и падает на не-ASCII символах. Принудительно UTF-8.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from ragbench.config import Config, make_backends
from ragbench.ingest import load_documents, chunk_documents
from ragbench.index import Index
from ragbench.retrieval import retrieve, MODES
from ragbench.answer import answer
from ragbench import evaluate


# --- индекс: собрать или поднять из кэша ---

def get_index(cfg: Config, emb, info, rebuild: bool = False, verbose: bool = True):
    want_tfidf = bool(getattr(emb, "has_state", False))
    if not rebuild and Index.exists(cfg.index_path):
        meta = Index.read_meta(cfg.index_path)
        cached_tfidf = meta.get("embedder", {}).get("kind") == "tfidf"
        same_chunk = (meta.get("chunk_size") == cfg.chunk_size
                      and meta.get("chunk_overlap") == cfg.chunk_overlap)
        if cached_tfidf == want_tfidf and same_chunk:
            if verbose:
                print(f"[index] поднимаю из кэша: {cfg.index_path} ({len(meta['chunks'])} чанков)")
            return Index.load(cfg.index_path, embedder=emb)
        if verbose:
            print("[index] кэш не подходит (сменился провайдер/чанкинг) — пересобираю")

    docs = load_documents(cfg.data_dir, cfg.doc_glob)
    if not docs:
        sys.exit(f"[index] в {cfg.data_dir} нет документов (*.txt/*.md/*.pdf)")
    chunks = chunk_documents(docs, cfg.chunk_size, cfg.chunk_overlap)
    if verbose:
        srcs = ", ".join(sorted({d["source"] for d in docs}))
        print(f"[index] собираю: {len(docs)} док. ({srcs}) -> {len(chunks)} чанков, "
              f"эмбеддер {info['embedder']}")
    idx = Index.build(chunks, emb)
    idx.save(cfg.index_path, extra={"chunk_size": cfg.chunk_size,
                                    "chunk_overlap": cfg.chunk_overlap})
    if verbose:
        print(f"[index] сохранил {cfg.index_path} (матрица {idx.vectors.shape})")
    return idx


def banner(info: dict):
    print(f"провайдер: {info['resolved']} | эмбеддер: {info['embedder']} | LLM: {info['llm']}")


def _print_sources(rr, conf=None, label=None):
    if conf is not None:
        print(f"\nуверенность: {conf:.3f} ({label})")
    for note in rr.notes:
        print(f"  · {note}")
    if rr.expansions:
        print("  расширение запроса:")
        for e in rr.expansions:
            print(f"    ~ {e[:120]}")
    print("источники:")
    for rank, h in enumerate(rr.hits, 1):
        prev = h.chunk.text.replace("\n", " ")[:80]
        sec = f" · {h.chunk.section}" if h.chunk.section else ""
        print(f"  {rank}. [#{h.id}{sec}] ({h.chunk.source})  «{prev}…»")


# --- команды ---

def cmd_info(cfg, emb, llm, info, args=None):
    banner(info)
    print(f"данные: {cfg.data_dir} | индекс: {cfg.index_path}")
    print(f"чанкинг: size={cfg.chunk_size} overlap={cfg.chunk_overlap} | top_k={cfg.top_k} "
          f"| pool={cfg.candidate_pool} | порог уверенности={cfg.min_similarity}")
    print(f"режимы retrieval: {', '.join(MODES)}")


def cmd_build(cfg, emb, llm, info, args):
    get_index(cfg, emb, info, rebuild=True)


def cmd_ask(cfg, emb, llm, info, args):
    banner(info)
    idx = get_index(cfg, emb, info)
    rr = retrieve(idx, args.question, mode=args.mode, k=cfg.top_k, llm=llm, pool=cfg.candidate_pool)
    a = answer(idx, args.question, rr, llm=llm, cfg=cfg)
    print(f"\nответ ({'LLM' if a.grounded else 'экстрактивно'}):\n{a.text}")
    _print_sources(rr, a.confidence, a.conf_label)


def cmd_repl(cfg, emb, llm, info, args):
    banner(info)
    idx = get_index(cfg, emb, info)
    print(f"\nРежим: {args.mode}. Вопрос — Enter, пустая строка — выход.")
    print("Примеры: адрес офиса? · сколько суточных в Москву? · можно ли работать из дома?")
    while True:
        try:
            q = input("\nвопрос> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nвыход."); break
        if not q:
            print("выход."); break
        rr = retrieve(idx, q, mode=args.mode, k=cfg.top_k, llm=llm, pool=cfg.candidate_pool)
        a = answer(idx, q, rr, llm=llm, cfg=cfg)
        print(f"\nответ: {a.text}")
        _print_sources(rr, a.confidence, a.conf_label)


def cmd_eval(cfg, emb, llm, info, args):
    banner(info)
    idx = get_index(cfg, emb, info)
    questions = evaluate.load_eval_set(os.path.join(cfg.data_dir, "eval_set.json"))
    row = evaluate.evaluate_mode(idx, questions, args.mode, llm, cfg)
    print(f"\neval [{args.mode}] на {len(questions)} вопросах:")
    print(evaluate.render_table([row]))


def cmd_bench(cfg, emb, llm, info, args):
    banner(info)
    idx = get_index(cfg, emb, info)
    questions = evaluate.load_eval_set(os.path.join(cfg.data_dir, "eval_set.json"))
    print(f"\nСравнение режимов на {len(questions)} вопросах "
          f"(Hit@1/Hit@3/MRR — retrieval, Ans — сквозной ответ):\n")
    rows = evaluate.run_bench(idx, questions, llm, cfg)
    print(evaluate.render_table(rows))
    out = os.path.join(cfg.data_dir, "eval_results.json")
    evaluate.save_results(rows, info, out)
    print(f"\nсохранено: {out}")


def cmd_web(cfg, emb, llm, info, args):
    from web import serve
    idx = get_index(cfg, emb, info)
    serve(idx, llm, cfg, info, port=args.port)


def main():
    p = argparse.ArgumentParser(description="RAG-стенд (ДЗ_4)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("info")
    b = sub.add_parser("build"); b.add_argument("--chunk", type=int); b.add_argument("--overlap", type=int)
    a = sub.add_parser("ask"); a.add_argument("question"); a.add_argument("--mode", default="hybrid", choices=MODES); a.add_argument("--k", type=int)
    r = sub.add_parser("repl"); r.add_argument("--mode", default="hybrid", choices=MODES)
    e = sub.add_parser("eval"); e.add_argument("--mode", default="hybrid", choices=MODES)
    sub.add_parser("bench")
    w = sub.add_parser("web"); w.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    cfg = Config()
    if getattr(args, "chunk", None):
        cfg.chunk_size = args.chunk
    if getattr(args, "overlap", None) is not None:
        cfg.chunk_overlap = args.overlap
    if getattr(args, "k", None):
        cfg.top_k = args.k

    emb, llm, info = make_backends(cfg)
    {"info": cmd_info, "build": cmd_build, "ask": cmd_ask, "repl": cmd_repl,
     "eval": cmd_eval, "bench": cmd_bench, "web": cmd_web}[args.cmd](cfg, emb, llm, info, args)


if __name__ == "__main__":
    main()
