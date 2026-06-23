"""Flat RAG baseline (Step 4 comparison).

A real vector retriever over the document chunks. To keep the lab runnable with
zero heavy downloads, similarity is computed with TF-IDF + cosine. Two real
backends:

    * scikit-learn TfidfVectorizer if installed (preferred).
    * a self-contained pure-NumPy TF-IDF otherwise.

This mirrors what a ChromaDB/FAISS flat index would return (top-k nearest
chunks), which is exactly the baseline the lab asks us to beat with GraphRAG.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import List, Optional

import numpy as np

from .config import Config
from .corpus import Chunk
from .llm import LLM


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP = set(
    "the a an and or of to in on for with at by from as is are was were be been "
    "this that these those it its their his her our your we they he she you i "
    "not no but if then than so such can will would could should may might".split()
)


def _tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP and len(t) > 1]


class _NumpyTfidf:
    """Minimal but real TF-IDF vectorizer (smooth-idf, l2-normalised)."""

    def __init__(self):
        self.vocab: dict = {}
        self.idf: Optional[np.ndarray] = None

    def fit_transform(self, docs: List[str]) -> np.ndarray:
        tokenised = [_tokenize(d) for d in docs]
        df = Counter()
        for toks in tokenised:
            for w in set(toks):
                df[w] += 1
        self.vocab = {w: i for i, w in enumerate(sorted(df))}
        n = len(docs)
        self.idf = np.zeros(len(self.vocab))
        for w, i in self.vocab.items():
            self.idf[i] = math.log((1 + n) / (1 + df[w])) + 1.0
        return self._vectorize(tokenised)

    def transform(self, docs: List[str]) -> np.ndarray:
        return self._vectorize([_tokenize(d) for d in docs])

    def _vectorize(self, tokenised: List[List[str]]) -> np.ndarray:
        mat = np.zeros((len(tokenised), len(self.vocab)))
        for r, toks in enumerate(tokenised):
            if not toks:
                continue
            tf = Counter(toks)
            for w, c in tf.items():
                j = self.vocab.get(w)
                if j is not None:
                    mat[r, j] = (c / len(toks)) * self.idf[j]
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms


class FlatRAG:
    """Top-k cosine retriever over chunks + LLM answer synthesis."""

    def __init__(self, cfg: Config, llm: Optional[LLM] = None):
        self.cfg = cfg
        self.llm = llm
        self.chunks: List[Chunk] = []
        self._matrix = None
        self._backend = None
        self._vec = None

    def index(self, chunks: List[Chunk]) -> "FlatRAG":
        self.chunks = chunks
        corpus = [f"{c.title}. {c.text}" for c in chunks]
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            self._vec = TfidfVectorizer(stop_words="english", max_features=20000)
            self._matrix = self._vec.fit_transform(corpus)
            self._backend = "sklearn"
        except Exception:
            self._vec = _NumpyTfidf()
            self._matrix = self._vec.fit_transform(corpus)
            self._backend = "numpy"
        return self

    def _cosine(self, query: str) -> np.ndarray:
        if self._backend == "sklearn":
            from sklearn.metrics.pairwise import cosine_similarity

            qv = self._vec.transform([query])
            return cosine_similarity(qv, self._matrix)[0]
        qv = self._vec.transform([query])          # (1, V), l2-normalised
        return (self._matrix @ qv[0])              # rows already l2-normalised

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Chunk]:
        k = top_k or self.cfg.flat_top_k
        sims = self._cosine(query)
        idx = np.argsort(-sims)[:k]
        return [self.chunks[i] for i in idx]

    def build_context(self, chunks: List[Chunk]) -> str:
        parts = []
        for c in chunks:
            parts.append(f"[{c.doc_id} | {c.title}] {c.text}")
        return "\n\n".join(parts)

    def answer(self, query: str, top_k: Optional[int] = None) -> dict:
        hits = self.retrieve(query, top_k)
        context = self.build_context(hits)
        system = (
            "You are a careful analyst. Answer the question using ONLY the "
            "provided context. If the answer is not in the context, say you "
            "don't have enough information. Be concise."
        )
        user = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        ans = self.llm.complete(system, user, stage="flat_rag_answer", max_tokens=400) if self.llm else context
        return {
            "system": "FlatRAG",
            "query": query,
            "answer": ans,
            "context_ids": [c.chunk_id for c in hits],
            "backend": self._backend,
        }
