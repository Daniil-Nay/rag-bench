"""
ingest.py — загрузка документов и нарезка на чанки.

Чем отличается от лекции: лекционный chunk_text режет текст вслепую каждые
N символов и рвёт предложения и таблицу суточных пополам. Здесь чанкинг
структурный — сначала разбиваем по разделам/абзацам, потом упаковываем
блоки в чанки ~chunk_size с перекрытием, стараясь не резать абзац посередине.
Каждый чанк помнит источник и название раздела — это идёт в цитату [#id].

Лекционное скользящее окно сохранено как sliding_window() и используется
как baseline (режим naive) и как fallback для очень длинных абзацев.
"""
from __future__ import annotations

import os
import re
import glob
from dataclasses import dataclass


@dataclass
class Chunk:
    id: int
    text: str
    source: str
    section: str = ""


_HEADING = re.compile(r"^\s*\d+\.\s+[А-ЯЁA-Z][А-ЯЁA-Z0-9 ,«»\-/]{2,}$")


# --- загрузка ---

def load_documents(data_dir: str, patterns: str = "*.txt,*.md") -> list[dict]:
    """Читает текстовые файлы (и .pdf, если доступен pypdf) -> [{source, text}]."""
    files: list[str] = []
    for pat in [p.strip() for p in patterns.split(",") if p.strip()]:
        files += glob.glob(os.path.join(data_dir, pat))
    files = sorted(set(files))

    docs: list[dict] = []
    for path in files:
        with open(path, encoding="utf-8") as f:
            docs.append({"source": os.path.basename(path), "text": f.read()})

    for path in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _read_pdf(path)
        if text:
            docs.append({"source": os.path.basename(path), "text": text})
    return docs


def _read_pdf(path: str) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        print(f"[ingest] пропускаю {os.path.basename(path)}: нет pypdf (pip install pypdf)")
        return ""
    reader = PdfReader(path)
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


# --- чанкинг ---

def sliding_window(text: str, size: int, overlap: int) -> list[str]:
    """Лекционный baseline: окно [start:start+size], шаг size-overlap."""
    out, start = [], 0
    step = max(1, size - overlap)
    while start < len(text):
        piece = text[start:start + size].strip()
        if piece:
            out.append(piece)
        start += step
    return out


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Режем документ на разделы по нумерованным заголовкам -> [(label, body)]."""
    sections: list[tuple[str, list[str]]] = []
    title, lines = "Преамбула", []
    for line in text.splitlines():
        if _HEADING.match(line.strip()):
            if lines:
                sections.append((title, lines))
            title = re.sub(r"^\s*\d+\.\s*", "", line.strip()).capitalize()
            lines = [line]
        else:
            lines.append(line)
    if lines:
        sections.append((title, lines))
    return [(t, "\n".join(ls).strip()) for t, ls in sections if "\n".join(ls).strip()]


def structure_chunks(text: str, size: int, overlap: int) -> list[tuple[str, str]]:
    """Раздел = чанк. Длинный раздел добиваем окном, повторяя заголовок -> [(text, label)]."""
    out: list[tuple[str, str]] = []
    for label, body in _split_sections(text):
        if len(body) <= size:
            out.append((body, label))
        else:
            head = body.splitlines()[0]
            for piece in sliding_window(body, size, overlap):
                tagged = piece if piece.startswith(head) else f"{head}\n{piece}"
                out.append((tagged, label))
    return out


def chunk_documents(docs: list[dict], size: int, overlap: int,
                    mode: str = "structure") -> list[Chunk]:
    """Превращает документы в плоский список Chunk с глобальными id."""
    out: list[Chunk] = []
    cid = 0
    for doc in docs:
        text, source = doc["text"], doc["source"]
        if mode == "naive":
            pieces = [(p, "") for p in sliding_window(text, size, overlap)]
        else:
            pieces = structure_chunks(text, size, overlap)
        for piece, section in pieces:
            out.append(Chunk(id=cid, text=piece, source=source, section=section))
            cid += 1
    return out
