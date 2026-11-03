"""
scorer.py — Hybrid scoring, ranking, filtering, and reasoning for job matching.

Responsibilities:
    1. hybrid_score()          — combine semantic (vector) + keyword (BM25) scores
    2. aggregate_candidate_scores() — pool chunk-level scores to resume-level
    3. build_bm25_index()      — build a BM25 index over retrieved chunk texts
    4. bm25_score_query()      — score chunks against a query with BM25
    5. parse_requirements()    — parse hard-filter strings like "5+ years Python"
    6. filter_candidates()     — apply hard filters to candidates
    7. build_reasoning()       — generate human-readable match explanation

Usage:
    See job_matcher.py for full integration example.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

# ---------------------------------------------------------------------------
# BM25 index
# ---------------------------------------------------------------------------


def build_bm25_index(corpus: list[str]):
    """Build a BM25 index over a list of text documents.

    Args:
        corpus: List of text strings (one per chunk).

    Returns:
        BM25Okapi instance from rank_bm25.
    """
    try:
        from rank_bm25 import BM25Okapi  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "rank_bm25 is required. Install with: pip install rank-bm25"
        ) from exc

    tokenized = [_tokenize(doc) for doc in corpus]
    return BM25Okapi(tokenized)


def bm25_score_query(bm25_index, query: str) -> list[float]:
    """Score all documents in the BM25 index against a query.

    Args:
        bm25_index: BM25Okapi instance.
        query: Query string.

    Returns:
        List of raw BM25 scores (one per document in the index).
    """
    tokens = _tokenize(query)
    scores = bm25_index.get_scores(tokens)
    return scores.tolist()


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer for BM25."""
    return re.findall(r"[a-z0-9]+", text.lower())


# ---------------------------------------------------------------------------
# Hybrid scoring
# ---------------------------------------------------------------------------


def hybrid_score(
    semantic_score: float,
    keyword_score: float,
    semantic_weight: float = 0.7,
    keyword_weight: float = 0.3,
) -> float:
    """Compute a weighted hybrid score from semantic and keyword scores.

    Args:
        semantic_score: Normalised semantic similarity score in [0, 1].
        keyword_score:  Normalised BM25 keyword score in [0, 1].
        semantic_weight: Weight for the semantic component (default 0.7).
        keyword_weight:  Weight for the keyword component (default 0.3).

    Returns:
        Final score scaled to [0, 100].
    """
    raw = semantic_weight * semantic_score + keyword_weight * keyword_score
    return round(min(max(raw * 100, 0), 100), 2)


def normalise_scores(scores: list[float]) -> list[float]:
    """Min-max normalise a list of scores to [0, 1].

    If all scores are identical, returns a list of zeros.
    """
    if not scores:
        return []
    min_s = min(scores)
    max_s = max(scores)
    if max_s == min_s:
        return [0.0] * len(scores)
    return [(s - min_s) / (max_s - min_s) for s in scores]


# ---------------------------------------------------------------------------
# Score aggregation across chunks
# ---------------------------------------------------------------------------


def aggregate_candidate_scores(
    chunk_results: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Aggregate chunk-level match results into resume-level candidates.

    Args:
        chunk_results: List of dicts, each representing one matched chunk.
            Required keys per dict:
                filename        (str)   — resume file basename
                path            (str)   — full path to resume file
                name            (str)   — candidate name
                section         (str)   — chunk's section label
                content         (str)   — chunk text
                hybrid_score    (float) — score 0–100 for this chunk
                matched_skills  (list)  — skills found in this chunk
                skills          (list)  — all candidate skills
                experience_years (int)  — from metadata
                education       (str)   — from metadata

    Returns:
        Dict keyed by filename → aggregated candidate dict.
    """
    candidates: dict[str, dict[str, Any]] = {}

    for chunk in chunk_results:
        fname = chunk["filename"]
        score = chunk.get("hybrid_score", 0.0)

        if fname not in candidates:
            candidates[fname] = {
                "candidate_name": chunk.get("name", "Unknown"),
                "resume_path": chunk.get("path", fname),
                "match_score": score,
                "best_chunk_score": score,
                "chunk_scores": [score],
                "matched_skills": set(chunk.get("matched_skills", [])),
                "all_skills": chunk.get("skills", []),
                "relevant_excerpts": [],
                "sections_matched": set(),
                "experience_years": chunk.get("experience_years", 0),
                "education": chunk.get("education", ""),
            }
        else:
            candidates[fname]["chunk_scores"].append(score)
            candidates[fname]["matched_skills"].update(chunk.get("matched_skills", []))
            if score > candidates[fname]["best_chunk_score"]:
                candidates[fname]["best_chunk_score"] = score

        # Collect top relevant excerpts (up to 3 per candidate)
        if (
            len(candidates[fname]["relevant_excerpts"]) < 3
            and chunk.get("content", "").strip()
        ):
            excerpt = chunk["content"].strip()[:300]
            if excerpt not in candidates[fname]["relevant_excerpts"]:
                candidates[fname]["relevant_excerpts"].append(excerpt)

        section = chunk.get("section", "")
        if section:
            candidates[fname]["sections_matched"].add(section)

    # Finalise: use best chunk score as match_score, convert sets to lists
    for fname, cand in candidates.items():
        chunk_scores = cand["chunk_scores"]
        # Weighted combination: 60% max score + 40% average score
        best = max(chunk_scores)
        avg = sum(chunk_scores) / len(chunk_scores)
        cand["match_score"] = round(0.6 * best + 0.4 * avg, 2)
        cand["matched_skills"] = sorted(cand["matched_skills"])
        cand["sections_matched"] = sorted(cand["sections_matched"])
        del cand["chunk_scores"]
        del cand["best_chunk_score"]

    return candidates


# ---------------------------------------------------------------------------
# Hard-requirement filtering
# ---------------------------------------------------------------------------

# Pattern: "N+ years <Skill>" e.g. "5+ years Python", "3+ years Machine Learning"
_YEARS_REQ_REGEX = re.compile(
    r"(\d+)\+?\s*years?\s+(.+)", re.IGNORECASE
)

# Pattern: "must have <Skill>" e.g. "must have Docker"
_MUST_HAVE_REGEX = re.compile(
    r"must\s+have\s+(.+)", re.IGNORECASE
)


def parse_requirements(requirements: list[str]) -> list[dict[str, Any]]:
    """Parse human-readable requirement strings into structured filters.

    Supported formats:
        "5+ years Python"          → {type: "years", skill: "python", min_years: 5}
        "must have Docker"         → {type: "skill", skill: "docker"}
        "Python"                   → {type: "skill", skill: "python"}

    Args:
        requirements: List of requirement strings.

    Returns:
        List of parsed requirement dicts.
    """
    parsed: list[dict[str, Any]] = []
    for req in requirements:
        req = req.strip()
        m_years = _YEARS_REQ_REGEX.match(req)
        m_must = _MUST_HAVE_REGEX.match(req)

        if m_years:
            parsed.append(
                {
                    "type": "years",
                    "skill": m_years.group(2).strip().lower(),
                    "min_years": int(m_years.group(1)),
                    "raw": req,
                }
            )
        elif m_must:
            parsed.append(
                {
                    "type": "skill",
                    "skill": m_must.group(1).strip().lower(),
                    "raw": req,
                }
            )
        else:
            parsed.append({"type": "skill", "skill": req.lower(), "raw": req})

    return parsed


def filter_candidates(
    candidates: list[dict[str, Any]],
    requirements: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Remove candidates that don't meet hard requirements.

    Args:
        candidates: List of aggregated candidate dicts.
        requirements: Parsed requirements from parse_requirements().

    Returns:
        Filtered list of candidates.
    """
    if not requirements:
        return candidates

    filtered: list[dict[str, Any]] = []
    for cand in candidates:
        skills_lower = {s.lower() for s in cand.get("all_skills", [])}
        exp_years = cand.get("experience_years", 0)
        passes = True

        for req in requirements:
            if req["type"] == "skill":
                skill_req = req["skill"]
                # Check if required skill is in candidate's skills
                if not any(skill_req in s for s in skills_lower):
                    passes = False
                    break

            elif req["type"] == "years":
                skill_req = req["skill"]
                min_years = req["min_years"]
                # Check skill presence AND experience threshold
                has_skill = any(skill_req in s for s in skills_lower)
                has_years = exp_years >= min_years
                if not (has_skill and has_years):
                    passes = False
                    break

        if passes:
            filtered.append(cand)

    return filtered


# ---------------------------------------------------------------------------
# Reasoning generation
# ---------------------------------------------------------------------------


def build_reasoning(
    candidate: dict[str, Any],
    jd_skills: list[str],
) -> str:
    """Generate a human-readable reasoning string for a candidate's match.

    Args:
        candidate: Aggregated candidate dict.
        jd_skills: List of skills extracted from the job description.

    Returns:
        A short paragraph explaining the match.
    """
    name = candidate.get("candidate_name", "Candidate")
    score = candidate.get("match_score", 0)
    matched = candidate.get("matched_skills", [])
    sections = candidate.get("sections_matched", [])
    exp_years = candidate.get("experience_years", 0)
    education = candidate.get("education", "")

    # Determine match quality label
    if score >= 80:
        quality = "strong"
    elif score >= 60:
        quality = "good"
    elif score >= 40:
        quality = "moderate"
    else:
        quality = "weak"

    parts: list[str] = [
        f"{name} is a {quality} match (score: {score}/100)."
    ]

    if matched:
        top_skills = matched[:5]
        parts.append(
            f"Matched skills: {', '.join(top_skills)}"
            + (f" and {len(matched) - 5} more." if len(matched) > 5 else ".")
        )

    if sections:
        parts.append(f"Relevant sections: {', '.join(sections)}.")

    if exp_years > 0:
        parts.append(f"Estimated {exp_years} year(s) of professional experience.")

    if education and education != "Not specified":
        parts.append(f"Education: {education[:80]}.")

    # Highlight any JD skills that are missing
    missing_skills = [s for s in jd_skills if s.lower() not in
                      {m.lower() for m in candidate.get("all_skills", [])}]
    if missing_skills:
        parts.append(
            f"Gap: {', '.join(missing_skills[:3])} not found in profile."
        )

    return " ".join(parts)
