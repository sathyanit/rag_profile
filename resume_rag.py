"""
resume_rag.py — Part A: Resume ingestion, chunking, embedding, and vector store.

Pipeline:
    1. Discover resume files (.pdf, .docx, .txt) in the specified directory.
    2. Load and extract text from each file.
    3. Chunk each resume into section-aware pieces.
    4. Extract structured metadata (name, skills, experience, education).
    5. Embed each chunk using sentence-transformers (local, no API key needed).
    6. Upsert into a persistent ChromaDB collection.

Usage:
    python resume_rag.py --resumes-dir resumes/
    python resume_rag.py --resumes-dir resumes/ --reset   # clear & re-ingest
    python resume_rag.py --resumes-dir resumes/ --model "all-MiniLM-L6-v2"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm
from colorama import Fore, Style, init as colorama_init

# Local utils
from utils.pdf_loader import load_resume, discover_resumes
from utils.chunker import chunk_resume
from utils.metadata import extract_metadata

# ---------------------------------------------------------------------------
# Initialise environment
# ---------------------------------------------------------------------------

load_dotenv()
colorama_init(autoreset=True)

# ---------------------------------------------------------------------------
# Constants / defaults (can be overridden by .env or CLI flags)
# ---------------------------------------------------------------------------

DEFAULT_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
DEFAULT_CHROMA_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
DEFAULT_COLLECTION = os.getenv("CHROMA_COLLECTION_NAME", "resumes")


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------


def load_embedding_model(model_name: str):
    """Load a SentenceTransformer embedding model.

    Args:
        model_name: HuggingFace model identifier, e.g. "all-MiniLM-L6-v2".

    Returns:
        SentenceTransformer model instance.
    """
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required. "
            "Install with: pip install sentence-transformers"
        ) from exc

    print(f"{Fore.CYAN}Loading embedding model: {model_name}{Style.RESET_ALL}")
    model = SentenceTransformer(model_name)
    print(f"{Fore.GREEN}✓ Model loaded.{Style.RESET_ALL}")
    return model


def embed_texts(model, texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a list of texts.

    Args:
        model: SentenceTransformer model.
        texts: List of strings to embed.

    Returns:
        List of embedding vectors (each is a list of floats).
    """
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embeddings.tolist()


# ---------------------------------------------------------------------------
# ChromaDB helpers
# ---------------------------------------------------------------------------


def get_chroma_collection(chroma_path: str, collection_name: str, reset: bool = False):
    """Connect to (or create) a ChromaDB collection.

    Args:
        chroma_path:     Path to the ChromaDB persistent directory.
        collection_name: Name of the collection.
        reset:           If True, delete the collection before recreating it.

    Returns:
        ChromaDB Collection instance.
    """
    try:
        import chromadb  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "chromadb is required. Install with: pip install chromadb"
        ) from exc

    client = chromadb.PersistentClient(path=chroma_path)

    if reset:
        try:
            client.delete_collection(collection_name)
            print(f"{Fore.YELLOW}⚠ Existing collection '{collection_name}' deleted.{Style.RESET_ALL}")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},  # cosine distance for semantic search
    )
    return collection


# ---------------------------------------------------------------------------
# Core ingestion pipeline
# ---------------------------------------------------------------------------


def ingest_resume(
    resume_path: Path,
    model,
    collection,
    verbose: bool = False,
) -> dict:
    """Full ingestion pipeline for a single resume file.

    Args:
        resume_path: Path to the resume file.
        model:       SentenceTransformer model for embeddings.
        collection:  ChromaDB collection to upsert into.
        verbose:     Print detailed chunk info.

    Returns:
        Summary dict with filename, chunk_count, metadata.
    """
    # Step 1: Load text
    doc = load_resume(resume_path)
    text = doc["text"]
    filename = doc["filename"]

    if not text.strip():
        print(f"  {Fore.YELLOW}⚠ Skipping {filename} — empty text.{Style.RESET_ALL}")
        return {"filename": filename, "chunk_count": 0, "skipped": True}

    # Step 2: Extract metadata
    meta = extract_metadata(text)

    # Step 3: Chunk the resume
    chunks = chunk_resume(text)

    if not chunks:
        print(f"  {Fore.YELLOW}⚠ Skipping {filename} — no chunks produced.{Style.RESET_ALL}")
        return {"filename": filename, "chunk_count": 0, "skipped": True}

    # Step 4: Build chunk texts and IDs
    chunk_texts = [c["content"] for c in chunks]
    chunk_ids = [f"{Path(filename).stem}_chunk_{c['chunk_index']}" for c in chunks]

    # Step 5: Embed all chunks in one batch call
    embeddings = embed_texts(model, chunk_texts)

    # Step 6: Build metadata for each chunk
    skills_json = json.dumps(meta["skills"])
    chunk_metadatas = [
        {
            "filename": filename,
            "path": str(resume_path),
            "section": c["section"],
            "chunk_index": c["chunk_index"],
            "name": meta["name"],
            "skills_json": skills_json,
            "experience_years": meta["experience_years"],
            "education": meta["education"][:200],  # ChromaDB metadata length limit
            "raw_skills_text": meta["raw_skills_text"][:500],
        }
        for c in chunks
    ]

    # Step 7: Upsert into ChromaDB
    collection.upsert(
        ids=chunk_ids,
        documents=chunk_texts,
        embeddings=embeddings,
        metadatas=chunk_metadatas,
    )

    if verbose:
        for c in chunks:
            print(
                f"    [{c['section']:20s}] chunk_{c['chunk_index']:02d} "
                f"({len(c['content'])} chars)"
            )

    return {
        "filename": filename,
        "name": meta["name"],
        "chunk_count": len(chunks),
        "skills_count": len(meta["skills"]),
        "experience_years": meta["experience_years"],
        "education": meta["education"],
        "skipped": False,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Resume RAG Ingestion Pipeline — embed resumes into ChromaDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python resume_rag.py --resumes-dir resumes/
  python resume_rag.py --resumes-dir sample_resumes/ --reset --verbose
  python resume_rag.py --resumes-dir resumes/ --model paraphrase-MiniLM-L6-v2
        """,
    )
    parser.add_argument(
        "--resumes-dir",
        default="./sample_resumes",
        help="Directory containing resume files (.pdf, .docx, .txt)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Sentence-transformer model name (default: {DEFAULT_MODEL})",
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
        "--reset",
        action="store_true",
        help="Delete existing collection before ingestion (full re-index)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-chunk details during ingestion",
    )

    args = parser.parse_args()

    # Banner
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"  Resume RAG Ingestion Pipeline")
    print(f"{'='*60}{Style.RESET_ALL}")

    # Discover resume files
    resumes_dir = Path(args.resumes_dir)
    try:
        resume_files = discover_resumes(resumes_dir)
    except FileNotFoundError as e:
        print(f"{Fore.RED}✗ {e}{Style.RESET_ALL}")
        sys.exit(1)

    if not resume_files:
        print(f"{Fore.YELLOW}No resume files found in: {resumes_dir}{Style.RESET_ALL}")
        sys.exit(0)

    print(f"\n{Fore.WHITE}Found {len(resume_files)} resume(s) in '{resumes_dir}'{Style.RESET_ALL}")

    # Load embedding model
    model = load_embedding_model(args.model)

    # Connect to ChromaDB
    print(f"\n{Fore.CYAN}Connecting to ChromaDB at: {args.chroma_path}{Style.RESET_ALL}")
    collection = get_chroma_collection(args.chroma_path, args.collection, reset=args.reset)
    print(f"{Fore.GREEN}✓ Collection '{args.collection}' ready.{Style.RESET_ALL}\n")

    # Ingest each resume
    results = []
    with tqdm(resume_files, desc="Ingesting resumes", unit="file") as pbar:
        for resume_path in pbar:
            pbar.set_postfix({"file": resume_path.name[:30]})
            try:
                result = ingest_resume(resume_path, model, collection, verbose=args.verbose)
                results.append(result)
            except Exception as exc:
                print(f"\n  {Fore.RED}✗ Error processing {resume_path.name}: {exc}{Style.RESET_ALL}")
                results.append({"filename": resume_path.name, "error": str(exc)})

    # Summary
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"  Ingestion Summary")
    print(f"{'='*60}{Style.RESET_ALL}")

    successful = [r for r in results if not r.get("skipped") and "error" not in r]
    skipped = [r for r in results if r.get("skipped")]
    errors = [r for r in results if "error" in r]

    for r in successful:
        print(
            f"  {Fore.GREEN}✓{Style.RESET_ALL} {r['filename']:<35} "
            f"name={r.get('name','?'):<20} "
            f"chunks={r.get('chunk_count',0):>3}  "
            f"skills={r.get('skills_count',0):>3}  "
            f"exp={r.get('experience_years',0)}yr"
        )

    for r in skipped:
        print(f"  {Fore.YELLOW}⚠{Style.RESET_ALL} {r['filename']:<35} SKIPPED (empty)")

    for r in errors:
        print(f"  {Fore.RED}✗{Style.RESET_ALL} {r['filename']:<35} ERROR: {r['error']}")

    total_chunks = sum(r.get("chunk_count", 0) for r in successful)
    print(f"\n{Fore.WHITE}Total:{Style.RESET_ALL} "
          f"{len(successful)} processed, "
          f"{len(skipped)} skipped, "
          f"{len(errors)} errors  |  "
          f"{Fore.CYAN}{total_chunks} chunks{Style.RESET_ALL} stored in ChromaDB\n")

    print(f"{Fore.GREEN}✓ Run job_matcher.py to search for candidates!{Style.RESET_ALL}\n")


if __name__ == "__main__":
    main()
