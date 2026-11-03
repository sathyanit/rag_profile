"""
chunker.py — Section-aware resume chunking.

Strategy:
    1. Detect section headers (EDUCATION, EXPERIENCE, SKILLS, etc.) using regex.
    2. Split the resume text into named sections.
    3. If a section exceeds MAX_TOKENS, split further at sentence boundaries.
    4. Always produce at least one chunk (the full text) even if no headers found.

Usage:
    from utils.chunker import chunk_resume

    chunks = chunk_resume(text)
    # chunks = [
    #   {"section": "SUMMARY",    "content": "...", "chunk_index": 0},
    #   {"section": "EXPERIENCE", "content": "...", "chunk_index": 1},
    #   ...
    # ]
"""

from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Approximate token limit per chunk (1 token ≈ 4 chars for English text)
MAX_CHARS_PER_CHUNK = 1800  # ~450 tokens — fits all-MiniLM-L6-v2 (512 token limit)

# Section header patterns — ordered by priority (more specific first)
SECTION_PATTERNS = [
    r"(?i)^(PROFESSIONAL\s+SUMMARY|SUMMARY|OBJECTIVE|PROFILE|ABOUT\s+ME)",
    r"(?i)^(WORK\s+EXPERIENCE|EXPERIENCE|EMPLOYMENT|CAREER\s+HISTORY|PROFESSIONAL\s+EXPERIENCE)",
    r"(?i)^(EDUCATION|ACADEMIC\s+BACKGROUND|QUALIFICATIONS)",
    r"(?i)^(SKILLS|TECHNICAL\s+SKILLS|CORE\s+COMPETENCIES|EXPERTISE|KEY\s+SKILLS)",
    r"(?i)^(PROJECTS|KEY\s+PROJECTS|PERSONAL\s+PROJECTS|SIDE\s+PROJECTS)",
    r"(?i)^(CERTIFICATIONS|CERTIFICATES|LICENSES|CREDENTIALS)",
    r"(?i)^(AWARDS|HONORS|ACHIEVEMENTS|ACCOMPLISHMENTS)",
    r"(?i)^(PUBLICATIONS|RESEARCH|PAPERS)",
    r"(?i)^(LANGUAGES|SPOKEN\s+LANGUAGES)",
    r"(?i)^(VOLUNTEER|VOLUNTEERING|COMMUNITY\s+SERVICE)",
    r"(?i)^(REFERENCES|REFEREES)",
    r"(?i)^(CONTACT|CONTACT\s+INFORMATION|PERSONAL\s+DETAILS)",
    r"(?i)^(INTERNSHIP|INTERNSHIPS|TRAINING)",
    r"(?i)^(HOBBIES|INTERESTS|EXTRACURRICULAR)",
]

# Combined pattern to detect any section header on its own line
_HEADER_REGEX = re.compile(
    r"^(" + "|".join(p.lstrip("(?i)^") for p in SECTION_PATTERNS) + r")\s*[:\-]?\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# Canonical section names (normalised)
def _canonical(header: str) -> str:
    """Normalise a detected section header to a canonical name."""
    h = header.strip().upper()
    mapping = {
        "PROFESSIONAL SUMMARY": "SUMMARY",
        "OBJECTIVE": "SUMMARY",
        "PROFILE": "SUMMARY",
        "ABOUT ME": "SUMMARY",
        "WORK EXPERIENCE": "EXPERIENCE",
        "EMPLOYMENT": "EXPERIENCE",
        "CAREER HISTORY": "EXPERIENCE",
        "PROFESSIONAL EXPERIENCE": "EXPERIENCE",
        "ACADEMIC BACKGROUND": "EDUCATION",
        "QUALIFICATIONS": "EDUCATION",
        "TECHNICAL SKILLS": "SKILLS",
        "CORE COMPETENCIES": "SKILLS",
        "EXPERTISE": "SKILLS",
        "KEY SKILLS": "SKILLS",
        "KEY PROJECTS": "PROJECTS",
        "PERSONAL PROJECTS": "PROJECTS",
        "SIDE PROJECTS": "PROJECTS",
        "CERTIFICATES": "CERTIFICATIONS",
        "LICENSES": "CERTIFICATIONS",
        "CREDENTIALS": "CERTIFICATIONS",
        "HONORS": "AWARDS",
        "ACHIEVEMENTS": "AWARDS",
        "ACCOMPLISHMENTS": "AWARDS",
        "RESEARCH": "PUBLICATIONS",
        "PAPERS": "PUBLICATIONS",
        "SPOKEN LANGUAGES": "LANGUAGES",
        "VOLUNTEERING": "VOLUNTEER",
        "COMMUNITY SERVICE": "VOLUNTEER",
        "REFEREES": "REFERENCES",
        "CONTACT INFORMATION": "CONTACT",
        "PERSONAL DETAILS": "CONTACT",
        "INTERNSHIPS": "INTERNSHIP",
        "TRAINING": "INTERNSHIP",
        "INTERESTS": "HOBBIES",
        "EXTRACURRICULAR": "HOBBIES",
    }
    return mapping.get(h, h)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def chunk_resume(text: str) -> list[dict[str, Any]]:
    """Split resume text into section-aware chunks.

    Args:
        text: Full raw text of a resume.

    Returns:
        List of chunk dicts, each with:
            section      (str)  — canonical section name or "HEADER" / "OTHER"
            content      (str)  — chunk text
            chunk_index  (int)  — 0-based sequential index
    """
    if not text or not text.strip():
        return []

    sections = _split_into_sections(text)
    chunks: list[dict[str, Any]] = []
    idx = 0

    for section_name, section_text in sections:
        sub_chunks = _split_long_section(section_text)
        for sub in sub_chunks:
            if sub.strip():
                chunks.append(
                    {
                        "section": section_name,
                        "content": sub.strip(),
                        "chunk_index": idx,
                    }
                )
                idx += 1

    # Fallback: if nothing chunked (no section headers found), treat whole text as one chunk
    if not chunks:
        chunks = [{"section": "FULL_TEXT", "content": text.strip(), "chunk_index": 0}]

    return chunks


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _split_into_sections(text: str) -> list[tuple[str, str]]:
    """Detect section headers and split text accordingly.

    Returns list of (canonical_section_name, section_text) tuples.
    The first segment (before any header) is labelled "HEADER".
    """
    matches = list(_HEADER_REGEX.finditer(text))

    if not matches:
        return [("FULL_TEXT", text)]

    sections: list[tuple[str, str]] = []

    # Text before the first header
    pre_header = text[: matches[0].start()].strip()
    if pre_header:
        sections.append(("HEADER", pre_header))

    for i, match in enumerate(matches):
        header_raw = match.group(0).strip().rstrip(":-").strip()
        section_name = _canonical(header_raw)

        # Content runs from end of this header to start of next (or end of text)
        content_start = match.end()
        content_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[content_start:content_end].strip()

        if content:
            sections.append((section_name, content))

    return sections


def _split_long_section(text: str) -> list[str]:
    """Split a section into smaller chunks if it exceeds MAX_CHARS_PER_CHUNK.

    Splits at sentence boundaries ('. ', '! ', '? ', or newlines).
    """
    if len(text) <= MAX_CHARS_PER_CHUNK:
        return [text]

    # Split at sentence boundaries
    sentence_endings = re.compile(r"(?<=[.!?])\s+|\n{2,}")
    sentences = sentence_endings.split(text)

    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(current) + len(sentence) + 1 <= MAX_CHARS_PER_CHUNK:
            current = (current + " " + sentence).strip()
        else:
            if current:
                chunks.append(current)
            # If a single sentence itself is too long, force-split by chars
            if len(sentence) > MAX_CHARS_PER_CHUNK:
                for start in range(0, len(sentence), MAX_CHARS_PER_CHUNK):
                    chunks.append(sentence[start : start + MAX_CHARS_PER_CHUNK])
                current = ""
            else:
                current = sentence

    if current:
        chunks.append(current)

    return chunks if chunks else [text]
