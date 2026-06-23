"""GraphRAG querying (Step 3).

Pipeline for a question:
    1. Extract the main entity/entities from the question.
    2. Link them to nodes in the knowledge graph (exact + fuzzy).
    3. BFS-traverse the neighbourhood up to `hops` (default 2).
    4. Textualise the visited triples into a context paragraph.
    5. Send context + question to the LLM (or return context offline).

The contrast with Flat RAG: Flat RAG returns whole text chunks ranked by lexical
similarity; GraphRAG returns *structured paths* of relations, so multi-hop
questions ("who founded the company that makes X") can be answered by walking
edges instead of hoping a single chunk contains every hop.
"""

from __future__ import annotations

import re
from collections import deque
from typing import Dict, List, Optional, Tuple

import networkx as nx

from .config import Config
from .corpus import Chunk
from .extract import find_entities
from .graph_store import canonicalize
from .llm import LLM


class GraphRAG:
    def __init__(
        self,
        G: nx.MultiDiGraph,
        cfg: Config,
        llm: Optional[LLM] = None,
        chunks: Optional[List[Chunk]] = None,
    ):
        self.G = G
        self.cfg = cfg
        self.llm = llm
        # lowercase index for fast linking
        self._lower = {n.lower(): n for n in G.nodes()}
        # chunk_id -> Chunk, for source-provenance retrieval (real GraphRAG: the
        # graph indexes WHERE a fact came from, so we can pull that text back)
        self._chunks: Dict[str, Chunk] = {c.chunk_id: c for c in (chunks or [])}

    # ---- entity linking ----------------------------------------------------
    def link_entities(self, question: str) -> List[str]:
        """Map question text to graph nodes (exact, canonical, then fuzzy)."""
        found = find_entities(question)
        linked: List[str] = []
        seen = set()

        def add(node: str):
            if node and node not in seen:
                seen.add(node)
                linked.append(node)

        for surface in found:
            canon = canonicalize(surface)
            if surface in self.G:
                add(surface)
            elif canon in self.G:
                add(canon)
            elif surface.lower() in self._lower:
                add(self._lower[surface.lower()])
            elif canon.lower() in self._lower:
                add(self._lower[canon.lower()])
            else:
                # fuzzy: substring match against node names
                for low, orig in self._lower.items():
                    if surface.lower() in low or low in surface.lower():
                        if abs(len(low) - len(surface)) <= 6:
                            add(orig)
                            break

        # last resort: token overlap with node names
        if not linked:
            q_tokens = set(re.findall(r"[a-z0-9]+", question.lower()))
            for low, orig in self._lower.items():
                if set(low.split()) & q_tokens and len(low) > 2:
                    add(orig)
                    if len(linked) >= 3:
                        break
        return linked

    # ---- traversal ---------------------------------------------------------
    def traverse(self, seeds: List[str], hops: Optional[int] = None) -> List[Tuple[str, str, str, str]]:
        """BFS up to `hops` from each seed over the UNDIRECTED neighbourhood.

        Returns a list of (subject, relation, object, evidence) triples on the
        visited frontier. We treat edges as undirected for reachability but keep
        the original direction + relation in the emitted triple.
        """
        hops = hops or self.cfg.graph_hops
        UG = self.G  # MultiDiGraph; we look at successors + predecessors
        collected: List[Tuple[str, str, str, str]] = []
        visited = set(seeds)
        self._last_visited = set(seeds)          # for provenance retrieval
        frontier = deque((s, 0) for s in seeds if s in self.G)

        while frontier and len(collected) < self.cfg.graph_max_neighbors:
            node, depth = frontier.popleft()
            if depth >= hops:
                continue
            # outgoing edges
            for _, nbr, d in UG.out_edges(node, data=True):
                collected.append((node, d.get("relation", "REL"), nbr, d.get("evidence", "")))
                if nbr not in visited:
                    visited.add(nbr)
                    self._last_visited.add(nbr)
                    frontier.append((nbr, depth + 1))
            # incoming edges
            for src, _, d in UG.in_edges(node, data=True):
                collected.append((src, d.get("relation", "REL"), node, d.get("evidence", "")))
                if src not in visited:
                    visited.add(src)
                    self._last_visited.add(src)
                    frontier.append((src, depth + 1))

        # de-dup triples preserving order
        out, seen = [], set()
        for tr in collected:
            key = (tr[0], tr[1], tr[2])
            if key not in seen:
                seen.add(key)
                out.append(tr)
        return out[: self.cfg.graph_max_neighbors]

    # ---- provenance: pull source-chunk text for the visited subgraph -------
    def provenance_chunks(
        self, seeds: List[str], question: str = "", max_chunks: int = 5
    ) -> List[Chunk]:
        """Return source chunks reachable from the seed entities, reranked by
        lexical relevance to the question (hybrid graph + text retrieval, as in
        production GraphRAG "local search").

        Step 1: restrict the candidate pool to chunks that the seed nodes (and
        their 1-hop neighbours) were extracted from -- this is the *graph* filter
        that Flat RAG does not have.
        Step 2: among those graph-connected chunks, rank by question overlap so a
        hub entity (e.g. Tesla, appearing in many chunks) still returns the chunk
        that actually answers the question rather than an arbitrary one.
        """
        if not self._chunks:
            return []
        graph_score: Dict[str, int] = {}
        nodes = set(seeds)
        # entity-name expansion: a hub seed ("Tesla") should also reach more
        # specific co-referent nodes ("Tesla Model 3") so their source chunks
        # enter the candidate pool.
        for s in seeds:
            sl = s.lower()
            for n in self.G.nodes():
                nl = n.lower()
                if n != s and (sl in nl.split() or (len(sl) > 3 and sl in nl)):
                    nodes.add(n)
        for s in list(nodes):
            if s in self.G:
                nodes.update(list(self.G.successors(s))[:25])
                nodes.update(list(self.G.predecessors(s))[:25])
        for n in nodes:
            if n not in self.G:
                continue
            weight = 3 if n in seeds else 1
            for cid in self.G.nodes[n].get("sources", []) or []:
                graph_score[cid] = graph_score.get(cid, 0) + weight

        q_terms = set(re.findall(r"[a-z0-9]+", question.lower()))
        q_terms = {t for t in q_terms if len(t) > 2}

        ranked = []
        for cid, gscore in graph_score.items():
            ch = self._chunks.get(cid)
            if ch is None:
                continue
            c_terms = set(re.findall(r"[a-z0-9]+", ch.text.lower()))
            lexical = len(q_terms & c_terms)
            # combine: graph connectivity (anchor) + lexical match (specificity)
            combined = lexical + 0.5 * gscore
            ranked.append((combined, gscore, cid))
        ranked.sort(key=lambda x: (-x[0], -x[1]))
        return [self._chunks[cid] for *_, cid in ranked[:max_chunks]]

    # ---- textualisation ----------------------------------------------------
    def textualize(self, triples: List[Tuple[str, str, str, str]], with_evidence: bool = True) -> str:
        if not triples:
            return "(no related facts found in the knowledge graph)"
        lines = []
        for s, r, o, ev in triples:
            rel = r.replace("_", " ").lower()
            lines.append(f"- {s} {rel} {o}.")
        return "\n".join(lines)

    def build_context(self, triples, prov_chunks) -> str:
        """Graph context = structured facts (subgraph) + evidence/source text."""
        parts = ["## Knowledge-graph facts (subgraph paths):", self.textualize(triples)]
        # supporting evidence sentences carried on the edges
        ev = [e for *_, e in triples if e]
        if ev:
            uniq, seen = [], set()
            for s in ev:
                k = s[:80]
                if k not in seen:
                    seen.add(k)
                    uniq.append(s)
            parts.append("\n## Supporting evidence (edge provenance):")
            parts.append("\n".join(f"- {s}" for s in uniq[:12]))
        # full source chunks reached through the graph (provenance retrieval)
        if prov_chunks:
            parts.append("\n## Source passages (graph provenance):")
            for c in prov_chunks:
                parts.append(f"[{c.doc_id} | {c.title}] {c.text}")
        return "\n".join(parts)

    # ---- end-to-end answer -------------------------------------------------
    def answer(self, question: str, hops: Optional[int] = None) -> dict:
        seeds = self.link_entities(question)
        triples = self.traverse(seeds, hops)
        prov = self.provenance_chunks(seeds, question=question, max_chunks=self.cfg.flat_top_k)
        context = self.build_context(triples, prov)
        system = (
            "You are a careful analyst. Answer the question using ONLY the "
            "provided knowledge-graph facts and source passages. If they are "
            "insufficient, say so. Be concise."
        )
        user = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        ans = self.llm.complete(system, user, stage="graph_rag_answer", max_tokens=400) if self.llm else context
        return {
            "system": "GraphRAG",
            "query": question,
            "answer": ans,
            "seeds": seeds,
            "n_facts": len(triples),
            "n_prov_chunks": len(prov),
            "facts": context,
        }
