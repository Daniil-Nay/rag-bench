"""ragbench — мини-платформа RAG на чистом numpy (ДЗ_4).

Пакетный layout:
  config.py     — env-конфиг и выбор провайдера (LM Studio / GitHub Models / offline)
  providers.py  — TF-IDF эмбеддер (offline) и API-клиенты (embeddings + chat)
  ingest.py     — загрузка txt/md/pdf и структурный чанкинг
  index.py      — dense + BM25 индекс с сохранением на диск
  retrieval.py  — режимы naive/bm25/hybrid/hyde/multiquery/rerank
  answer.py     — grounded-ответ, цитаты, уверенность, guardrail, стриминг
  evaluate.py   — метрики Hit@k/MRR и side-by-side сравнение режимов
"""
from .config import Config, make_backends, resolve_provider
from .ingest import load_documents, chunk_documents
from .index import Index
from .retrieval import retrieve, MODES
from .answer import answer, stream_answer

__all__ = ["Config", "make_backends", "resolve_provider", "load_documents",
           "chunk_documents", "Index", "retrieve", "MODES", "answer", "stream_answer"]
