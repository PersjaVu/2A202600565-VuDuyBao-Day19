"""Corpus loading + chunking for the Tech Company Corpus.

Each document in the dataset has the shape:

    Query: ...
    Title: ...
    Link: ...
    Snippet: ...

    Full Content:
    <body ...>

We parse those light headers (useful metadata) and keep the body for
extraction / retrieval.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from .config import Config


@dataclass
class Document:
    doc_id: str
    title: str
    link: str
    query: str
    text: str          # full content (capped)
    path: str


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str


_HEADER_RE = {
    "query": re.compile(r"^Query:\s*(.*)$", re.MULTILINE),
    "title": re.compile(r"^Title:\s*(.*)$", re.MULTILINE),
    "link": re.compile(r"^Link:\s*(.*)$", re.MULTILINE),
}


def _grab(pattern: re.Pattern, text: str) -> str:
    m = pattern.search(text)
    return m.group(1).strip() if m else ""


def load_corpus(cfg: Config) -> List[Document]:
    """Read every doc_*.txt under cfg.corpus_dir into Document objects."""
    root = Path(cfg.corpus_dir)
    if not root.exists():
        raise FileNotFoundError(f"Corpus directory not found: {root}")

    files = sorted(
        root.glob("doc_*.txt"),
        key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)),
    )
    docs: List[Document] = []
    for fp in files:
        raw = fp.read_text(encoding="utf-8", errors="ignore")
        title = _grab(_HEADER_RE["title"], raw)
        link = _grab(_HEADER_RE["link"], raw)
        query = _grab(_HEADER_RE["query"], raw)

        # body = everything after "Full Content:" if present, else whole file
        if "Full Content:" in raw:
            body = raw.split("Full Content:", 1)[1]
        else:
            body = raw
        body = body.strip()[: cfg.max_chars_per_doc]

        docs.append(
            Document(
                doc_id=fp.stem,
                title=title or fp.stem,
                link=link,
                query=query,
                text=body,
                path=str(fp),
            )
        )
    return docs


def _split_text(text: str, size: int, overlap: int) -> List[str]:
    """Sentence-aware sliding window over characters."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        # try to break on a sentence boundary near the end
        if end < n:
            window = text[start:end]
            cut = max(window.rfind(". "), window.rfind("? "), window.rfind("! "))
            if cut > size * 0.5:
                end = start + cut + 1
        chunks.append(text[start:end].strip())
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]


def chunk_documents(docs: List[Document], cfg: Config) -> List[Chunk]:
    """Break each document into overlapping retrieval chunks."""
    chunks: List[Chunk] = []
    for d in docs:
        for i, piece in enumerate(_split_text(d.text, cfg.chunk_size, cfg.chunk_overlap)):
            chunks.append(
                Chunk(
                    chunk_id=f"{d.doc_id}::chunk_{i}",
                    doc_id=d.doc_id,
                    title=d.title,
                    text=piece,
                )
            )
    return chunks
