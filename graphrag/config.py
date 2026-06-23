"""Central configuration for the GraphRAG lab.

Everything is overridable through environment variables so the same code runs
in three modes:

    * OFFLINE  (default) -> heuristic extraction + TF-IDF retrieval, no keys.
    * OPENAI            -> set OPENAI_API_KEY  (and optionally GRAPHRAG_LLM=openai).
    * ANTHROPIC         -> set ANTHROPIC_API_KEY (GRAPHRAG_LLM=anthropic).

The corpus path defaults to the dataset folder shipped with the lab but can be
pointed anywhere with GRAPHRAG_CORPUS.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _first_existing(*candidates: str) -> str:
    for c in candidates:
        if c and Path(c).exists():
            return str(Path(c).resolve())
    # fall back to the first candidate even if missing (clear error later)
    return str(Path(candidates[0]).resolve())


@dataclass
class Config:
    # --- data ---------------------------------------------------------------
    corpus_dir: str = field(
        default_factory=lambda: os.environ.get(
            "GRAPHRAG_CORPUS",
            _first_existing(
                r"c:\Users\ADMIN\Downloads\dataset\dataset",
                "dataset",
                "../dataset/dataset",
            ),
        )
    )
    out_dir: str = field(default_factory=lambda: os.environ.get("GRAPHRAG_OUT", "outputs"))

    # cap very large documents so offline extraction stays fast (doc_50 is 3MB)
    max_chars_per_doc: int = 40_000
    chunk_size: int = 900          # characters per chunk (Flat RAG units)
    chunk_overlap: int = 150

    # --- extraction ---------------------------------------------------------
    # "heuristic" (offline, default) | "openai" | "anthropic"
    extractor: str = field(default_factory=lambda: os.environ.get("GRAPHRAG_EXTRACTOR", "heuristic"))
    max_chunks_for_llm_extract: int = 60   # safety cap on paid extraction

    # --- LLM (answer generation) -------------------------------------------
    # "none" (offline, just returns context) | "openai" | "anthropic"
    llm_backend: str = field(default_factory=lambda: os.environ.get("GRAPHRAG_LLM", "none"))
    openai_model: str = field(default_factory=lambda: os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    anthropic_model: str = field(
        default_factory=lambda: os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
    )

    # --- retrieval ----------------------------------------------------------
    flat_top_k: int = 4
    graph_hops: int = 2
    graph_max_neighbors: int = 40   # cap textualised neighbours per query

    # --- misc ---------------------------------------------------------------
    seed: int = 42

    def ensure_out(self) -> Path:
        p = Path(self.out_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def describe(self) -> str:
        return (
            f"corpus_dir = {self.corpus_dir}\n"
            f"out_dir    = {self.out_dir}\n"
            f"extractor  = {self.extractor}\n"
            f"llm_backend= {self.llm_backend}\n"
            f"chunk_size = {self.chunk_size} (overlap {self.chunk_overlap})\n"
        )
