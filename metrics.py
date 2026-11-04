"""
metrics.py — Performance evaluation and latency benchmarking for Resume RAG.

Metrics implemented:
    1. Precision@K (P@K)   — Proportion of top-K retrieved candidates that are relevant
    2. Recall@K (R@K)      — Proportion of all relevant candidates captured in top-K
    3. nDCG@K (nDCG@K)     — Normalized Discounted Cumulative Gain at K
    4. Query Latency       — Breakdown across embedding, vector retrieval, BM25 & ranking
    5. Ingestion Latency   — Per-resume and total ingestion throughput
"""

from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LatencyResult:
    embed_ms: float
    retrieval_ms: float
    ranking_ms: float
    total_ms: float
    top_k: int
    n_retrieved_chunks: int


# ---------------------------------------------------------------------------
# Retrieval Accuracy Metrics
# ---------------------------------------------------------------------------

def compute_precision_at_k(
    ranked_list: Sequence[str],
    relevant_set: set[str] | Sequence[str],
    k: int,
) -> float:
    """Compute Precision@K.

    Precision@K = (number of relevant items in top K) / K

    Args:
        ranked_list: List of candidate identifiers in ranked order.
        relevant_set: Set of ground-truth relevant candidate identifiers.
        k: Cutoff rank.

    Returns:
        Float value between 0.0 and 1.0.
    """
    if k <= 0:
        return 0.0
    relevant_set = set(relevant_set)
    top_k = ranked_list[:k]
    num_hits = sum(1 for item in top_k if item in relevant_set)
    return round(num_hits / k, 4)


def compute_recall_at_k(
    ranked_list: Sequence[str],
    relevant_set: set[str] | Sequence[str],
    k: int,
) -> float:
    """Compute Recall@K.

    Recall@K = (number of relevant items in top K) / (total relevant items)

    Args:
        ranked_list: List of candidate identifiers in ranked order.
        relevant_set: Set of ground-truth relevant candidate identifiers.
        k: Cutoff rank.

    Returns:
        Float value between 0.0 and 1.0.
    """
    relevant_set = set(relevant_set)
    if not relevant_set:
        return 0.0
    top_k = ranked_list[:k]
    num_hits = sum(1 for item in top_k if item in relevant_set)
    return round(num_hits / len(relevant_set), 4)


def compute_ndcg_at_k(
    ranked_list: Sequence[str],
    relevant_set: set[str] | Sequence[str],
    k: int,
) -> float:
    """Compute Normalized Discounted Cumulative Gain at K (nDCG@K) for binary relevance.

    DCG@K = sum_{i=1}^K [ rel_i / log2(i + 1) ]
    IDCG@K = sum_{i=1}^{min(K, |relevant|)} [ 1 / log2(i + 1) ]
    nDCG@K = DCG@K / IDCG@K

    Args:
        ranked_list: List of candidate identifiers in ranked order.
        relevant_set: Set of ground-truth relevant candidate identifiers.
        k: Cutoff rank.

    Returns:
        Float value between 0.0 and 1.0.
    """
    if k <= 0:
        return 0.0
    relevant_set = set(relevant_set)
    if not relevant_set:
        return 0.0

    # DCG
    dcg = 0.0
    for i, item in enumerate(ranked_list[:k]):
        rel = 1.0 if item in relevant_set else 0.0
        dcg += rel / math.log2(i + 2)

    # Ideal DCG
    ideal_hits = min(k, len(relevant_set))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

    if idcg == 0.0:
        return 0.0
    return round(dcg / idcg, 4)


# ---------------------------------------------------------------------------
# Latency Profiling
# ---------------------------------------------------------------------------

def measure_query_latency(
    jd_text: str,
    model: Any,
    collection: Any,
    n_runs: int = 10,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Measure latency decomposed into Embedding, Vector Retrieval, and Ranking.

    Args:
        jd_text: Job description string.
        model: SentenceTransformer model.
        collection: ChromaDB collection.
        n_runs: Number of benchmark iterations.
        top_k: Top K candidates to retrieve.

    Returns:
        List of dicts containing latency breakdown per iteration in milliseconds.
    """
    from utils.metadata import _extract_skills
    from utils.scorer import (
        build_bm25_index,
        bm25_score_query,
        normalise_scores,
        hybrid_score,
        aggregate_candidate_scores,
        build_reasoning,
    )
    import json

    results: list[dict[str, Any]] = []

    # Warm-up run (not recorded)
    _ = model.encode([jd_text[:100]])

    for _ in range(n_runs):
        t_start = time.perf_counter()

        # Stage 1: Embedding
        t_embed_0 = time.perf_counter()
        jd_embedding = model.encode([jd_text], convert_to_numpy=True)[0].tolist()
        embed_ms = (time.perf_counter() - t_embed_0) * 1000

        # Stage 2: ChromaDB Retrieval
        t_ret_0 = time.perf_counter()
        over_k = top_k * 3
        total_docs = collection.count()
        n_results = min(over_k, total_docs) if total_docs > 0 else 0
        chroma_res = collection.query(
            query_embeddings=[jd_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        retrieval_ms = (time.perf_counter() - t_ret_0) * 1000

        # Stage 3: BM25 + Hybrid Ranking + Aggregation
        t_rank_0 = time.perf_counter()
        raw_docs = chroma_res["documents"][0] if chroma_res["documents"] else []
        raw_metas = chroma_res["metadatas"][0] if chroma_res["metadatas"] else []
        raw_distances = chroma_res["distances"][0] if chroma_res["distances"] else []
        semantic_similarities = [1.0 - (d / 2.0) for d in raw_distances]

        if raw_docs:
            bm25_index = build_bm25_index(raw_docs)
            bm25_raw = bm25_score_query(bm25_index, jd_text)
            bm25_norm = normalise_scores(bm25_raw)
            sem_norm = normalise_scores(semantic_similarities)
            jd_skills = _extract_skills(jd_text)

            chunk_results = []
            for doc, meta, sem, kw in zip(raw_docs, raw_metas, sem_norm, bm25_norm):
                h_score = hybrid_score(sem, kw)
                chunk_skills = _extract_skills(doc)
                matched_in_chunk = [s for s in jd_skills if s.lower() in {cs.lower() for cs in chunk_skills}]
                try:
                    all_skills = json.loads(meta.get("skills_json", "[]"))
                except Exception:
                    all_skills = []

                chunk_results.append({
                    "filename": meta.get("filename", "unknown"),
                    "path": meta.get("path", ""),
                    "name": meta.get("name", "Unknown"),
                    "section": meta.get("section", ""),
                    "content": doc,
                    "hybrid_score": h_score,
                    "matched_skills": matched_in_chunk,
                    "skills": all_skills,
                    "experience_years": int(meta.get("experience_years", 0)),
                    "education": meta.get("education", ""),
                })

            candidates_map = aggregate_candidate_scores(chunk_results)
            candidates_list = sorted(candidates_map.values(), key=lambda c: c["match_score"], reverse=True)
            _ = candidates_list[:top_k]

        ranking_ms = (time.perf_counter() - t_rank_0) * 1000
        total_ms = (time.perf_counter() - t_start) * 1000

        results.append({
            "embed_ms": round(embed_ms, 2),
            "retrieval_ms": round(retrieval_ms, 2),
            "ranking_ms": round(ranking_ms, 2),
            "total_ms": round(total_ms, 2),
            "top_k": top_k,
            "n_retrieved_chunks": len(raw_docs),
        })

    return results


def measure_ingestion_latency(
    resume_paths: list[Path],
    model: Any,
    collection: Any,
) -> dict[str, Any]:
    """Benchmark end-to-end ingestion latency across a batch of resume files."""
    from resume_rag import ingest_resume

    file_latencies = []
    t_start = time.perf_counter()

    for path in resume_paths:
        t0 = time.perf_counter()
        res = ingest_resume(path, model, collection, verbose=False)
        duration_ms = (time.perf_counter() - t0) * 1000
        file_latencies.append({
            "file": path.name,
            "latency_ms": round(duration_ms, 2),
            "chunks": res.get("chunk_count", 0),
        })

    total_time_s = time.perf_counter() - t_start
    total_chunks = sum(f["chunks"] for f in file_latencies)

    return {
        "total_files": len(resume_paths),
        "total_chunks": total_chunks,
        "total_duration_seconds": round(total_time_s, 2),
        "avg_ms_per_file": round(float(np.mean([f["latency_ms"] for f in file_latencies])), 2) if file_latencies else 0.0,
        "p50_ms_per_file": round(float(np.percentile([f["latency_ms"] for f in file_latencies], 50)), 2) if file_latencies else 0.0,
        "p95_ms_per_file": round(float(np.percentile([f["latency_ms"] for f in file_latencies], 95)), 2) if file_latencies else 0.0,
        "throughput_files_per_sec": round(len(resume_paths) / max(total_time_s, 1e-4), 2),
        "per_file": file_latencies,
    }


def benchmark_all_jds(
    jd_dict: dict[str, str],
    ground_truth: dict[str, list[str]],
    model: Any,
    collection: Any,
    k_values: list[int] = [1, 3, 5, 10],
) -> list[dict[str, Any]]:
    """Compute complete benchmark evaluation across all job descriptions."""
    from job_matcher import match_jobs

    report = []
    for jd_name, jd_text in jd_dict.items():
        if jd_name not in ground_truth:
            continue

        relevant_set = set(ground_truth[jd_name])
        t0 = time.perf_counter()
        match_res = match_jobs(jd_text, model, collection, top_k=max(k_values))
        duration_ms = (time.perf_counter() - t0) * 1000

        ranked_stems = [Path(m["resume_path"]).stem for m in match_res["top_matches"]]

        entry: dict[str, Any] = {
            "jd_file": jd_name,
            "jd_label": jd_name.replace("jd_", "").replace(".txt", "").replace("_", " ").title(),
            "latency_ms": round(duration_ms, 1),
            "total_candidates": match_res["total_candidates_evaluated"],
            "ranked_matches": ranked_stems,
        }

        for k in k_values:
            entry[f"P@{k}"] = compute_precision_at_k(ranked_stems, relevant_set, k)
            entry[f"R@{k}"] = compute_recall_at_k(ranked_stems, relevant_set, k)
            entry[f"nDCG@{k}"] = compute_ndcg_at_k(ranked_stems, relevant_set, k)

        report.append(entry)

    return report


if __name__ == "__main__":
    print("metrics.py module loaded successfully. Ready for evaluation and benchmarks.")
