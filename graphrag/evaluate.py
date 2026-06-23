"""Benchmark + evaluation (Step 4).

Runs the 20-question benchmark on BOTH systems and scores each answer by gold
keyword recall (fraction of reference keywords found in the returned answer +
retrieved context). This is a real, automatic, system-agnostic metric:

    recall = |gold_keywords found| / |gold_keywords|

A "hallucination" flag is raised for Flat RAG when it returns a confident answer
(non-empty, no "don't have enough information") but misses the gold facts while
GraphRAG gets them right -- the exact case the lab asks us to log.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List

from .flat_rag import FlatRAG
from .graph_rag import GraphRAG


def load_questions(path: str) -> List[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["questions"] if isinstance(data, dict) else data


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower())


def keyword_recall(text: str, keywords: List[str]) -> float:
    if not keywords:
        return 0.0
    t = _norm(text)
    hit = sum(1 for k in keywords if _norm(k) in t)
    return hit / len(keywords)


_REFUSAL = re.compile(r"don't have enough|not enough information|no .* found|cannot find|insufficient", re.I)


def run_benchmark(questions: List[dict], flat: FlatRAG, graph: GraphRAG) -> List[dict]:
    rows = []
    for q in questions:
        kws = q["keywords"]

        fr = flat.answer(q["question"])
        flat_text = f"{fr['answer']}\n{flat.build_context(flat.retrieve(q['question']))}"
        flat_recall = keyword_recall(flat_text, kws)

        gr = graph.answer(q["question"])
        graph_text = f"{gr['answer']}\n{gr['facts']}"
        graph_recall = keyword_recall(graph_text, kws)

        flat_confident = bool(fr["answer"].strip()) and not _REFUSAL.search(fr["answer"])
        # hallucination: Flat answers confidently but misses facts while Graph nails them
        flat_halluc = flat_confident and flat_recall < 0.5 and graph_recall >= 0.5

        rows.append(
            {
                "id": q["id"],
                "type": q["type"],
                "question": q["question"],
                "flat_recall": round(flat_recall, 3),
                "graph_recall": round(graph_recall, 3),
                "winner": _winner(flat_recall, graph_recall),
                "flat_hallucination": flat_halluc,
                "flat_answer": _short(fr["answer"]),
                "graph_answer": _short(gr["answer"]),
                "graph_seeds": ", ".join(gr.get("seeds", [])),
                "reference": q.get("reference", ""),
            }
        )
    return rows


def _winner(f: float, g: float) -> str:
    if abs(f - g) < 1e-6:
        return "tie"
    return "GraphRAG" if g > f else "FlatRAG"


def _short(s: str, n: int = 200) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s[:n] + ("..." if len(s) > n else "")


def summarize(rows: List[dict]) -> dict:
    n = len(rows)
    flat_avg = sum(r["flat_recall"] for r in rows) / n if n else 0
    graph_avg = sum(r["graph_recall"] for r in rows) / n if n else 0
    return {
        "n_questions": n,
        "flat_avg_recall": round(flat_avg, 3),
        "graph_avg_recall": round(graph_avg, 3),
        "graph_wins": sum(1 for r in rows if r["winner"] == "GraphRAG"),
        "flat_wins": sum(1 for r in rows if r["winner"] == "FlatRAG"),
        "ties": sum(1 for r in rows if r["winner"] == "tie"),
        "flat_hallucinations": sum(1 for r in rows if r["flat_hallucination"]),
    }


def save_results(rows: List[dict], summary: dict, out_dir: str) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # CSV (pandas if available, else stdlib csv)
    csv_path = out / "benchmark_results.csv"
    cols = [
        "id", "type", "question", "flat_recall", "graph_recall", "winner",
        "flat_hallucination", "flat_answer", "graph_answer", "graph_seeds", "reference",
    ]
    try:
        import pandas as pd

        pd.DataFrame(rows)[cols].to_csv(csv_path, index=False, encoding="utf-8-sig")
    except Exception:
        import csv

        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in cols})

    (out / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {"csv": str(csv_path), "summary": str(out / "benchmark_summary.json")}
