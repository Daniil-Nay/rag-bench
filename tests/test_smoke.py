"""
Дымовые тесты RAG-стенда. Всё offline (TF-IDF), без сети и LM Studio.
Запуск:  pytest -q        или        python tests/test_smoke.py
"""
import os
import sys

os.environ.setdefault("RAG_PROVIDER", "offline")   # форсим offline до импорта конфигов
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragbench.config import Config, make_backends
from ragbench.ingest import load_documents, chunk_documents, sliding_window
from ragbench.index import Index
from ragbench.retrieval import retrieve, MODES
from ragbench.answer import answer
from ragbench import evaluate

CFG = Config()
EMB, LLM, INFO = make_backends(CFG)
DOCS = load_documents(CFG.data_dir, CFG.doc_glob)
CHUNKS = chunk_documents(DOCS, CFG.chunk_size, CFG.chunk_overlap)
IDX = Index.build(CHUNKS, EMB)
QS = evaluate.load_eval_set(os.path.join(CFG.data_dir, "eval_set.json"))


def test_provider_is_offline():
    assert INFO["resolved"] == "offline" and not LLM.has_llm


def test_chunking_has_sections():
    assert len(CHUNKS) >= 8
    labeled = [c for c in CHUNKS if c.section]
    assert len(labeled) >= len(CHUNKS) - 1            # все, кроме преамбулы
    assert all(c.text.strip() for c in CHUNKS)


def test_sliding_window_overlap():
    w = sliding_window("abcdefghij", size=4, overlap=2)
    assert w[0] == "abcd" and w[1] == "cdef"          # шаг = size-overlap = 2


def test_tfidf_query_dim_matches_index():
    q = IDX.embed_query("сколько суточных в Москву")
    assert q.shape[0] == IDX.vectors.shape[1]


def test_index_save_load_roundtrip(tmp_path=None):
    import tempfile
    path = os.path.join(tempfile.mkdtemp(), "idx.npz")
    IDX.save(path, extra={"chunk_size": CFG.chunk_size, "chunk_overlap": CFG.chunk_overlap})
    idx2 = Index.load(path)
    assert idx2.vectors.shape == IDX.vectors.shape
    assert len(idx2.chunks) == len(IDX.chunks)


def test_bm25_finds_rare_token():
    # уникальный токен @svp_parking должен поднять раздел про парковку первым
    hits = retrieve(IDX, "как забронировать парковку svp_parking", mode="bm25", k=3, llm=LLM).hits
    assert any("svp_parking" in h.chunk.text for h in hits[:1])


def test_all_modes_return_k_hits():
    for m in MODES:
        rr = retrieve(IDX, "сколько дней отпуска", mode=m, k=3, llm=LLM, pool=CFG.candidate_pool)
        assert 1 <= len(rr.hits) <= 3, m


def test_guardrail_on_out_of_doc():
    rr = retrieve(IDX, "кто генеральный директор и какая выручка", mode="hybrid", k=3, llm=LLM)
    a = answer(IDX, "кто генеральный директор и какая выручка", rr, llm=LLM, cfg=CFG)
    assert "ответа нет" in a.text.lower()


def test_retrieval_quality_floor():
    # на gold-наборе hybrid не должен проседать ниже разумного порога
    row = evaluate.evaluate_mode(IDX, QS, "hybrid", LLM, CFG)
    assert row["hit3"] >= 0.8, row
    assert row["ans"] >= 0.8, row


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  ok  {name}")
            except AssertionError as e:
                fails += 1
                print(f"FAIL  {name}: {e}")
    print("итог:", "всё зелёное" if not fails else f"{fails} провал(ов)")
    sys.exit(1 if fails else 0)
