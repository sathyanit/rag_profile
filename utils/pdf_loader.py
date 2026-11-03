"""
pdf_loader.py — Extract raw text from resume files.

Supported formats:
    .pdf   → pdfplumber (layout-aware, handles multi-column)
    .docx  → python-docx (paragraph-level extraction)
    .txt   → plain read (UTF-8 with fallback to latin-1)

Usage:
    from utils.pdf_loader import load_resume

    doc = load_resume("resumes/john_doe.pdf")
    # doc = {"text": "...", "path": "resumes/john_doe.pdf",
    #         "filename": "john_doe.pdf", "format": "pdf", "pages": 2}
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def load_resume(path: str | Path) -> dict[str, Any]:
    """Load a resume file and return its text plus metadata.

    Args:
        path: Absolute or relative path to a .pdf, .docx, or .txt file.

    Returns:
        dict with keys:
            text      (str)  — full extracted text
            path      (str)  — original file path as string
            filename  (str)  — basename of the file
            format    (str)  — "pdf" | "docx" | "txt"
            pages     (int)  — number of pages (PDFs) or 1 for others

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file extension is not supported.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Resume file not found: {path}")

    ext = path.suffix.lower()
    if ext == ".pdf":
        return _load_pdf(path)
    elif ext == ".docx":
        return _load_docx(path)
    elif ext == ".txt":
        return _load_txt(path)
    else:
        raise ValueError(
            f"Unsupported file format '{ext}'. Supported: .pdf, .docx, .txt"
        )


# ---------------------------------------------------------------------------
# Internal loaders
# ---------------------------------------------------------------------------


def _load_pdf(path: Path) -> dict[str, Any]:
    """Extract text from a PDF using pdfplumber."""
    try:
        import pdfplumber  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "pdfplumber is required for PDF parsing. "
            "Install it with: pip install pdfplumber"
        ) from exc

    pages_text: list[str] = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text.strip())

    full_text = "\n\n".join(pages_text)
    return {
        "text": full_text,
        "path": str(path),
        "filename": path.name,
        "format": "pdf",
        "pages": len(pages_text),
    }


def _load_docx(path: Path) -> dict[str, Any]:
    """Extract text from a DOCX file using python-docx."""
    try:
        from docx import Document  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "python-docx is required for DOCX parsing. "
            "Install it with: pip install python-docx"
        ) from exc

    doc = Document(str(path))
    paragraphs = [para.text.strip() for para in doc.paragraphs if para.text.strip()]
    full_text = "\n".join(paragraphs)
    return {
        "text": full_text,
        "path": str(path),
        "filename": path.name,
        "format": "docx",
        "pages": 1,
    }


def _load_txt(path: Path) -> dict[str, Any]:
    """Read a plain-text resume file."""
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="latin-1")

    return {
        "text": text.strip(),
        "path": str(path),
        "filename": path.name,
        "format": "txt",
        "pages": 1,
    }


def discover_resumes(directory: str | Path) -> list[Path]:
    """Walk a directory and return all resume file paths.

    Args:
        directory: Path to the folder containing resume files.

    Returns:
        Sorted list of Path objects for each supported resume file.
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Resume directory not found: {directory}")

    supported = {".pdf", ".docx", ".txt"}
    files = [
        p
        for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in supported
    ]
    return sorted(files)
