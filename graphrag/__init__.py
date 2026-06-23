"""GraphRAG lab package — Day 19.

Modular GraphRAG pipeline over the Tech Company Corpus:
    corpus -> triple extraction -> NetworkX knowledge graph
            -> Flat RAG  (TF-IDF / embeddings)
            -> GraphRAG  (entity-linking + 2-hop BFS traversal)
            -> benchmark + cost analysis.

All heavy / paid dependencies (OpenAI, Anthropic, sentence-transformers,
Neo4j, NodeRAG) are OPTIONAL. The pipeline runs fully offline using
deterministic heuristics + TF-IDF so the lab is reproducible without keys.
"""

from .config import Config
from .corpus import load_corpus, chunk_documents
from .extract import extract_triples, get_extractor
from .graph_store import build_graph, save_graph, load_graph, draw_graph
from .flat_rag import FlatRAG
from .graph_rag import GraphRAG
from .llm import get_llm, TokenMeter

__all__ = [
    "Config",
    "load_corpus",
    "chunk_documents",
    "extract_triples",
    "get_extractor",
    "build_graph",
    "save_graph",
    "load_graph",
    "draw_graph",
    "FlatRAG",
    "GraphRAG",
    "get_llm",
    "TokenMeter",
]
