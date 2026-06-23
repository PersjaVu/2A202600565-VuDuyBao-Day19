"""Knowledge-graph construction with NetworkX (Step 2, Choice A).

Responsibilities:
    * Canonicalise / DEDUPLICATE entities (the lab's "why dedup matters" point).
    * Build a directed multigraph: nodes = entities, edges = typed relations.
    * Persist / load the graph (GraphML + JSON node table).
    * Visualise with matplotlib (Deliverable #2: knowledge-graph screenshot).
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx

from .config import Config
from .extract import Triple


# --------------------------------------------------------------------------- #
# Entity canonicalisation / deduplication
# --------------------------------------------------------------------------- #
# Alias map -> canonical name. Without this, "GM", "General Motors" and
# "Alphabet"/"Google" become separate nodes and multi-hop traversal breaks.
_ALIASES = {
    "gm": "General Motors",
    "general motors": "General Motors",
    "alphabet": "Google",
    "google": "Google",
    "facebook": "Meta",
    "meta": "Meta",
    "lucid motors": "Lucid",
    "lucid": "Lucid",
    "mercedes": "Mercedes-Benz",
    "mercedes-benz": "Mercedes-Benz",
    "kelley blue book": "Kelley Blue Book",
    "kbb": "Kelley Blue Book",
    "lg energy solution": "LG Energy Solution",
    "the u.s.": "United States",
    "u.s.": "United States",
    "us": "United States",
    "united states": "United States",
    "ev": "electric vehicle",
}

_SUFFIX_RE = re.compile(r"[,]?\s+(Inc|Corp|Corporation|Co|Ltd|LLC|Group|Motors|Holdings)\.?$", re.I)


def canonicalize(name: str) -> str:
    """Map surface forms to a single canonical entity string."""
    n = re.sub(r"\s+", " ", name).strip(" .,:;'\"")
    key = n.lower()
    if key in _ALIASES:
        return _ALIASES[key]
    # strip corporate suffixes for dedup, but keep proper casing
    stripped = _SUFFIX_RE.sub("", n).strip()
    key2 = stripped.lower()
    if key2 in _ALIASES:
        return _ALIASES[key2]
    return stripped or n


def build_graph(triples: List[Triple], cfg: Optional[Config] = None) -> nx.MultiDiGraph:
    """Build a deduplicated directed knowledge graph from triples."""
    G = nx.MultiDiGraph()
    # edge weight = number of times a (subj, rel, obj) was seen (confidence)
    edge_count: Dict[Tuple[str, str, str], int] = defaultdict(int)
    edge_evidence: Dict[Tuple[str, str, str], str] = {}
    node_mentions: Dict[str, int] = defaultdict(int)
    node_sources: Dict[str, set] = defaultdict(set)

    for t in triples:
        s = canonicalize(t.subject)
        o = canonicalize(t.object)
        r = t.relation.strip().upper().replace(" ", "_")
        if not s or not o or s.lower() == o.lower():
            continue
        if len(s) > 60 or len(o) > 60:
            continue
        key = (s, r, o)
        edge_count[key] += 1
        if key not in edge_evidence and t.evidence:
            edge_evidence[key] = t.evidence
        node_mentions[s] += 1
        node_mentions[o] += 1
        if t.source_chunk:
            node_sources[s].add(t.source_chunk)
            node_sources[o].add(t.source_chunk)

    for node, cnt in node_mentions.items():
        G.add_node(node, mentions=cnt, sources=sorted(node_sources[node])[:20])

    for (s, r, o), w in edge_count.items():
        G.add_edge(s, o, key=r, relation=r, weight=w, evidence=edge_evidence.get((s, r, o), ""))

    return G


def graph_stats(G: nx.MultiDiGraph) -> dict:
    rels = defaultdict(int)
    for _, _, d in G.edges(data=True):
        rels[d.get("relation", "?")] += 1
    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "relation_types": dict(sorted(rels.items(), key=lambda x: -x[1])),
        "density": round(nx.density(G), 5),
    }


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def save_graph(G: nx.MultiDiGraph, out_dir: str) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # GraphML can't store list attributes -> stringify sources first
    H = G.copy()
    for _, data in H.nodes(data=True):
        if isinstance(data.get("sources"), list):
            data["sources"] = ";".join(data["sources"])
    graphml_path = out / "knowledge_graph.graphml"
    nx.write_graphml(H, graphml_path)

    # also a friendly JSON edge list
    edges = [
        {"subject": u, "relation": d.get("relation"), "object": v, "weight": d.get("weight", 1)}
        for u, v, d in G.edges(data=True)
    ]
    (out / "triples.json").write_text(json.dumps(edges, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"graphml": str(graphml_path), "triples_json": str(out / "triples.json")}


def load_graph(path: str) -> nx.MultiDiGraph:
    return nx.read_graphml(path)


# --------------------------------------------------------------------------- #
# Visualisation (Deliverable #2)
# --------------------------------------------------------------------------- #
def draw_graph(
    G: nx.MultiDiGraph,
    out_path: str,
    top_n: int = 60,
    title: str = "Tech Company Knowledge Graph (GraphRAG)",
) -> Optional[str]:
    """Draw the most-connected sub-graph to a PNG. Returns path or None."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # matplotlib optional
        print(f"[draw_graph] matplotlib unavailable ({e}); skipping PNG.")
        return None

    if G.number_of_nodes() == 0:
        print("[draw_graph] empty graph; nothing to draw.")
        return None

    # keep the top_n nodes by degree for legibility
    deg = dict(G.degree())
    keep = [n for n, _ in sorted(deg.items(), key=lambda x: -x[1])[:top_n]]
    Sub = G.subgraph(keep).copy()

    plt.figure(figsize=(18, 13))
    pos = nx.spring_layout(Sub, k=0.6, iterations=60, seed=42)
    sizes = [200 + 90 * Sub.degree(n) for n in Sub.nodes()]
    nx.draw_networkx_nodes(Sub, pos, node_size=sizes, node_color="#4C9BE8", alpha=0.85)
    nx.draw_networkx_edges(Sub, pos, alpha=0.25, edge_color="#888888", arrows=True, arrowsize=8)
    nx.draw_networkx_labels(Sub, pos, font_size=8, font_color="#111111")

    # label a subset of edges with their relation type
    edge_labels = {}
    for u, v, d in Sub.edges(data=True):
        if (u, v) not in edge_labels:
            edge_labels[(u, v)] = d.get("relation", "")
    nx.draw_networkx_edge_labels(Sub, pos, edge_labels=edge_labels, font_size=6, font_color="#B00020")

    plt.title(title, fontsize=16)
    plt.axis("off")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    return out_path
