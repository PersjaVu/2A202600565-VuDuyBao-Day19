#!/usr/bin/env python
"""
LAB DAY 19 - GraphRAG with the Tech Company Corpus  (script version)
====================================================================

End-to-end, REAL (non-mock) pipeline:

    Step 1  Indexing     : extract entities + relations -> triples
    Step 2  Construction : build a deduplicated NetworkX knowledge graph  (Choice A)
    Step 3  Querying     : entity-link + 2-hop BFS traversal + textualisation
    Step 4  Evaluation   : 20-question benchmark, Flat RAG vs GraphRAG + cost report

Run it:

    python graphrag_lab.py                      # full pipeline, offline
    python graphrag_lab.py --demo               # quick 1-query demo
    GRAPHRAG_LLM=anthropic python graphrag_lab.py   # use a real LLM for answers
    GRAPHRAG_EXTRACTOR=anthropic python graphrag_lab.py  # LLM triple extraction

Outputs land in ./outputs (graph PNG, GraphML, triples.json, benchmark CSV,
cost report).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from graphrag import (
    Config,
    FlatRAG,
    GraphRAG,
    TokenMeter,
    build_graph,
    chunk_documents,
    draw_graph,
    extract_triples,
    get_llm,
    load_corpus,
    save_graph,
)
from graphrag.extract import get_extractor
from graphrag.graph_store import graph_stats
from graphrag.evaluate import (
    load_questions,
    run_benchmark,
    save_results,
    summarize,
)


def banner(msg: str):
    print("\n" + "=" * 72)
    print(f"  {msg}")
    print("=" * 72)


def build_everything(cfg: Config, meter: TokenMeter):
    """Run Steps 1-2 and return (chunks, graph, flat, graph_rag, llm)."""
    out = cfg.ensure_out()
    print(cfg.describe())

    # ---- load ----
    banner("Loading corpus")
    docs = load_corpus(cfg)
    chunks = chunk_documents(docs, cfg)
    print(f"  documents : {len(docs)}")
    print(f"  chunks    : {len(chunks)}")

    llm = get_llm(cfg, meter)
    print(f"  llm backend (answers)   : {llm.backend} ({llm.model or 'n/a'})")
    extractor = get_extractor(cfg, llm)
    print(f"  extractor (indexing)    : {extractor.name}")

    # ---- Step 1: extraction ----
    banner("Step 1  -  Entity & Relation extraction (Indexing)")
    t0 = time.perf_counter()
    triples = extract_triples(chunks, cfg, llm)
    t_extract = time.perf_counter() - t0
    if extractor.name == "heuristic":
        # account offline extraction cost as token-equivalents over scanned text
        scanned = sum(len(c.text) for c in chunks)
        meter.add("extract", prompt=scanned // 4, completion=len(triples) * 12, seconds=t_extract)
    print(f"  raw triples extracted : {len(triples)}  ({t_extract:.2f}s)")
    for t in triples[:8]:
        print(f"    ({t.subject})  -[{t.relation}]->  ({t.object})")

    # ---- Step 2: graph construction ----
    banner("Step 2  -  Knowledge-graph construction (NetworkX, deduplicated)")
    t0 = time.perf_counter()
    G = build_graph(triples, cfg)
    t_build = time.perf_counter() - t0
    meter.add("graph_build", seconds=t_build)
    stats = graph_stats(G)
    print(f"  nodes={stats['nodes']}  edges={stats['edges']}  density={stats['density']}  ({t_build:.2f}s)")
    print("  top relation types:")
    for rel, c in list(stats["relation_types"].items())[:8]:
        print(f"    {rel:<18} {c}")

    paths = save_graph(G, str(out))
    print(f"  saved: {paths['graphml']}")
    png = draw_graph(G, str(out / "knowledge_graph.png"))
    if png:
        print(f"  saved: {png}")

    # ---- retrievers ----
    flat = FlatRAG(cfg, llm).index(chunks)
    graph_rag = GraphRAG(G, cfg, llm, chunks=chunks)
    print(f"  Flat RAG backend: {flat._backend}")

    (out / "graph_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return chunks, G, flat, graph_rag, llm


def demo(flat: FlatRAG, graph_rag: GraphRAG):
    banner("Step 3  -  Demo multi-hop query")
    q = "Which automaker cancelled plans to build affordable EVs with Honda and set a 2035 goal?"
    print(f"  Q: {q}\n")
    fr = flat.answer(q)
    gr = graph_rag.answer(q)
    print(f"  [FlatRAG]  -> {fr['answer'][:300]}")
    print(f"\n  [GraphRAG] seeds={gr['seeds']}  facts={gr['n_facts']}")
    print(f"  [GraphRAG] -> {gr['answer'][:300]}")


def run_eval(cfg: Config, flat: FlatRAG, graph_rag: GraphRAG, meter: TokenMeter):
    banner("Step 4  -  Benchmark: Flat RAG  vs  GraphRAG")
    qpath = Path("benchmark/questions.json")
    questions = load_questions(str(qpath))
    print(f"  questions: {len(questions)}")

    t0 = time.perf_counter()
    rows = run_benchmark(questions, flat, graph_rag)
    print(f"  ran benchmark in {time.perf_counter()-t0:.2f}s")

    summary = summarize(rows)
    paths = save_results(rows, summary, cfg.out_dir)

    print("\n  --- per-question (recall) ---")
    print(f"  {'id':<3} {'type':<7} {'flat':<6} {'graph':<6} {'winner':<9} halluc")
    for r in rows:
        print(
            f"  {r['id']:<3} {r['type']:<7} {r['flat_recall']:<6} {r['graph_recall']:<6} "
            f"{r['winner']:<9} {'YES' if r['flat_hallucination'] else ''}"
        )
    print("\n  --- summary ---")
    for k, v in summary.items():
        print(f"    {k:<22} {v}")
    print(f"\n  saved: {paths['csv']}")

    # cost report
    banner("Cost / Token usage  (Deliverable #4)")
    print(meter.report())
    Path(cfg.out_dir, "cost_report.txt").write_text(meter.report(), encoding="utf-8")
    Path(cfg.out_dir, "cost_report.json").write_text(
        json.dumps(meter.as_rows(), indent=2), encoding="utf-8"
    )
    return rows, summary


def main():
    ap = argparse.ArgumentParser(description="GraphRAG Lab Day 19")
    ap.add_argument("--demo", action="store_true", help="run a single demo query only")
    ap.add_argument("--no-eval", action="store_true", help="skip the 20-question benchmark")
    args = ap.parse_args()

    cfg = Config()
    meter = TokenMeter()

    _, _, flat, graph_rag, _ = build_everything(cfg, meter)

    if args.demo:
        demo(flat, graph_rag)
        return
    demo(flat, graph_rag)
    if not args.no_eval:
        run_eval(cfg, flat, graph_rag, meter)
    banner("DONE")


if __name__ == "__main__":
    main()
