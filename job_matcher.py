"""
job_matcher.py — Part B: Semantic job matching engine with hybrid search and ranking.

Pipeline:
    1. Load the persistent ChromaDB collection (ingested by resume_rag.py).
    2. Embed the input job description using the same sentence-transformer model.
    3. Over-retrieve top K*3 chunks via semantic search (cosine similarity).
    4. Build a BM25 index over the retrieved chunks for keyword matching.
    5. Compute hybrid scores (70% semantic + 30% BM25) per chunk.
    6. Aggregate chunk scores per resume (60% best + 40% average).
    7. Extract skills from the JD and identify which ones matched each candidate.
    8. Apply hard requirement filters (e.g. "5+ years Python").
    9. Return top-K candidates as structured JSON matching the assignment spec.

Usage:
    python job_matcher.py --jd "Senior Python ML Engineer..."
    python job_matcher.py --jd "..." --top-k 5 --require "5+ years Python" --require "TensorFlow"
    python job_matcher.py --jd-file job_description.txt --output results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from colorama import Fore, Style, init as colorama_init

from utils.metadata import extract_metadata, _extract_skills  # reuse skill extractor
from utils.scorer import (
    build_bm25_index,
    bm25_score_query,
    normalise_scores,
    hybrid_score,
    aggregate_candidate_scores,
    parse_requirements,
    filter_candidates,
    build_reasoning,
)

# ---------------------------------------------------------------------------
# Initialise environment
# ---------------------------------------------------------------------------

load_dotenv()
colorama_init(autoreset=True)

DEFAULT_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
DEFAULT_CHROMA_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
DEFAULT_COLLECTION = os.getenv("CHROMA_COLLECTION_NAME", "resumes")
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "10"))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_embedding_model(model_name: str):
    """Load the SentenceTransformer embedding model."""
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required. "
            "Install with: pip install sentence-transformers"
        ) from exc
    return SentenceTransformer(model_name)


def get_chroma_collection(chroma_path: str, collection_name: str):
    """Connect to an existing ChromaDB collection."""
    try:
        import chromadb  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "chromadb is required. Install with: pip install chromadb"
        ) from exc

    client = chromadb.PersistentClient(path=chroma_path)

    try:
        collection = client.get_collection(name=collection_name)
    except Exception:
        raise RuntimeError(
            f"Collection '{collection_name}' not found in ChromaDB at '{chroma_path}'.\n"
            "Run resume_rag.py first to ingest resumes."
        )
    return collection


def embed_query(model, text: str) -> list[float]:
    """Embed a single query string."""
    vec = model.encode([text], convert_to_numpy=True)
    return vec[0].tolist()


# ---------------------------------------------------------------------------
# Core matching pipeline
# ---------------------------------------------------------------------------


def match_jobs(
    jd_text: str,
    model,
    collection,
    top_k: int = DEFAULT_TOP_K,
    requirements: list[str] | None = None,
) -> dict:
    """Run the full job matching pipeline.

    Args:
        jd_text:      Job description text.
        model:        SentenceTransformer model (same as used during ingestion).
        collection:   ChromaDB collection of resume chunks.
        top_k:        Number of top candidates to return.
        requirements: List of hard-requirement strings (e.g. ["5+ years Python"]).

    Returns:
        Dict matching the assignment output spec.
    """
    requirements = requirements or []

    # -------------------------------------------------------------------------
    # Step 1: Embed the job description
    # -------------------------------------------------------------------------
    print(f"{Fore.CYAN}Step 1/6:{Style.RESET_ALL} Embedding job description...")
    jd_embedding = embed_query(model, jd_text)

    # Extract skills from the JD for matching and reasoning
    jd_skills = _extract_skills(jd_text)
    print(f"  JD skills detected: {', '.join(jd_skills[:10]) or 'none'}")

    # -------------------------------------------------------------------------
    # Step 2: Semantic search — retrieve top K*3 chunks
    # -------------------------------------------------------------------------
    over_k = top_k * 3
    print(f"\n{Fore.CYAN}Step 2/6:{Style.RESET_ALL} Semantic search (retrieving top {over_k} chunks)...")

    total_docs = collection.count()
    if total_docs == 0:
        print(f"{Fore.RED}✗ No documents in ChromaDB collection. Run resume_rag.py first.{Style.RESET_ALL}")
        return _empty_result(jd_text)

    n_results = min(over_k, total_docs)
    results = collection.query(
        query_embeddings=[jd_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    raw_docs: list[str] = results["documents"][0]
    raw_metas: list[dict] = results["metadatas"][0]
    # ChromaDB returns cosine distances [0,2]; convert to similarity [0,1]
    raw_distances: list[float] = results["distances"][0]
    semantic_similarities = [1.0 - (d / 2.0) for d in raw_distances]

    print(f"  Retrieved {len(raw_docs)} chunks from {len(set(m['filename'] for m in raw_metas))} resumes.")

    # -------------------------------------------------------------------------
    # Step 3: BM25 keyword scoring over retrieved chunks
    # -------------------------------------------------------------------------
    print(f"\n{Fore.CYAN}Step 3/6:{Style.RESET_ALL} BM25 keyword scoring...")
    bm25_index = build_bm25_index(raw_docs)
    bm25_raw_scores = bm25_score_query(bm25_index, jd_text)
    bm25_norm = normalise_scores(bm25_raw_scores)

    # Also normalise semantic scores for fair weighting
    sem_norm = normalise_scores(semantic_similarities)

    # -------------------------------------------------------------------------
    # Step 4: Hybrid score per chunk + identify matched skills
    # -------------------------------------------------------------------------
    print(f"\n{Fore.CYAN}Step 4/6:{Style.RESET_ALL} Computing hybrid scores...")
    chunk_results: list[dict] = []
    for i, (doc, meta, sem, kw) in enumerate(
        zip(raw_docs, raw_metas, sem_norm, bm25_norm)
    ):
        h_score = hybrid_score(sem, kw)

        # Skills matched in this specific chunk
        chunk_skills = _extract_skills(doc)
        matched_in_chunk = [s for s in jd_skills if s.lower() in
                            {cs.lower() for cs in chunk_skills}]

        # Deserialise skills_json stored in metadata
        try:
            all_skills = json.loads(meta.get("skills_json", "[]"))
        except (json.JSONDecodeError, TypeError):
            all_skills = []

        chunk_results.append(
            {
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
            }
        )

    # -------------------------------------------------------------------------
    # Step 5: Aggregate to resume level
    # -------------------------------------------------------------------------
    print(f"\n{Fore.CYAN}Step 5/6:{Style.RESET_ALL} Aggregating candidate scores...")
    candidates_map = aggregate_candidate_scores(chunk_results)
    candidates_list = list(candidates_map.values())

    # -------------------------------------------------------------------------
    # Step 6: Hard-requirement filtering
    # -------------------------------------------------------------------------
    if requirements:
        parsed_reqs = parse_requirements(requirements)
        print(f"\n{Fore.CYAN}Step 6/6:{Style.RESET_ALL} Applying {len(parsed_reqs)} hard filter(s)...")
        for r in parsed_reqs:
            print(f"  Filter: {r['raw']}")
        candidates_list = filter_candidates(candidates_list, parsed_reqs)
        print(f"  {len(candidates_list)} candidate(s) passed filters.")
    else:
        print(f"\n{Fore.CYAN}Step 6/6:{Style.RESET_ALL} No hard filters applied.")

    # -------------------------------------------------------------------------
    # Sort by match_score descending, then truncate to top_k
    # -------------------------------------------------------------------------
    candidates_list.sort(key=lambda c: c["match_score"], reverse=True)
    top_candidates = candidates_list[:top_k]

    # -------------------------------------------------------------------------
    # Build final output JSON
    # -------------------------------------------------------------------------
    top_matches = []
    for cand in top_candidates:
        reasoning = build_reasoning(cand, jd_skills)
        top_matches.append(
            {
                "candidate_name": cand["candidate_name"],
                "resume_path": cand["resume_path"],
                "match_score": cand["match_score"],
                "matched_skills": cand["matched_skills"],
                "relevant_excerpts": cand["relevant_excerpts"],
                "reasoning": reasoning,
                # Extra fields (bonus info)
                "experience_years": cand.get("experience_years", 0),
                "education": cand.get("education", ""),
                "sections_matched": cand.get("sections_matched", []),
            }
        )

    return {
        "job_description": jd_text,
        "top_matches": top_matches,
        "total_candidates_evaluated": len(candidates_map),
        "candidates_after_filter": len(candidates_list),
        "model_used": DEFAULT_MODEL,
    }


def _empty_result(jd_text: str) -> dict:
    return {"job_description": jd_text, "top_matches": [], "total_candidates_evaluated": 0}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Job Matching Engine — semantic + keyword resume search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python job_matcher.py --jd "Senior Python ML Engineer with TensorFlow experience"
  python job_matcher.py --jd "..." --top-k 5 --require "5+ years Python"
  python job_matcher.py --jd-file job_description.txt --output results.json
  python job_matcher.py --jd "..." --require "must have Docker" --require "AWS"
        """,
    )

    jd_group = parser.add_mutually_exclusive_group(required=True)
    jd_group.add_argument(
        "--jd",
        help="Job description as a string",
    )
    jd_group.add_argument(
        "--jd-file",
        help="Path to a text file containing the job description",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of top candidates to return (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--require",
        action="append",
        default=[],
        metavar="REQUIREMENT",
        help=(
            "Hard requirement to filter candidates. "
            "Supported formats: '5+ years Python', 'must have Docker', 'AWS'. "
            "Can be specified multiple times."
        ),
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Embedding model name (must match model used during ingestion, default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--chroma-path",
        default=DEFAULT_CHROMA_PATH,
        help=f"ChromaDB storage path (default: {DEFAULT_CHROMA_PATH})",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help=f"ChromaDB collection name (default: {DEFAULT_COLLECTION})",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save JSON results to a file",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        default=True,
        help="Pretty-print JSON output (default: True)",
    )

    args = parser.parse_args()

    # Load JD text
    if args.jd:
        jd_text = args.jd
    else:
        jd_path = Path(args.jd_file)
        if not jd_path.exists():
            print(f"{Fore.RED}✗ JD file not found: {jd_path}{Style.RESET_ALL}")
            sys.exit(1)
        jd_text = jd_path.read_text(encoding="utf-8").strip()

    # Banner
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"  Job Matching Engine")
    print(f"{'='*60}{Style.RESET_ALL}")
    print(f"\n{Fore.WHITE}Job Description:{Style.RESET_ALL}")
    print(f"  {jd_text[:200]}{'...' if len(jd_text) > 200 else ''}")
    print(f"\n{Fore.WHITE}Settings:{Style.RESET_ALL}")
    print(f"  top-k     : {args.top_k}")
    print(f"  model     : {args.model}")
    print(f"  chroma-db : {args.chroma_path}")
    print(f"  filters   : {args.require or 'none'}")
    print()

    # Load model
    model = load_embedding_model(args.model)

    # Connect to ChromaDB
    try:
        collection = get_chroma_collection(args.chroma_path, args.collection)
    except RuntimeError as e:
        print(f"\n{Fore.RED}✗ {e}{Style.RESET_ALL}\n")
        sys.exit(1)

    print(f"{Fore.GREEN}✓ Connected to ChromaDB ({collection.count()} chunks indexed).{Style.RESET_ALL}\n")

    # Run matching pipeline
    results = match_jobs(
        jd_text=jd_text,
        model=model,
        collection=collection,
        top_k=args.top_k,
        requirements=args.require,
    )

    # Output results
    indent = 2 if args.pretty else None
    json_output = json.dumps(results, indent=indent, ensure_ascii=False)

    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"  Results — Top {len(results['top_matches'])} Match(es)")
    print(f"{'='*60}{Style.RESET_ALL}\n")
    print(json_output)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json_output, encoding="utf-8")
        print(f"\n{Fore.GREEN}✓ Results saved to: {output_path}{Style.RESET_ALL}")

    print()


if __name__ == "__main__":
    main()
