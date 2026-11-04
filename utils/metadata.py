"""
metadata.py — Extract structured metadata fields from resume text.

Extracted fields:
    name              (str)  — candidate's full name (best guess)
    skills            (list) — list of detected technical/soft skills
    experience_years  (int)  — total years of professional experience
    education         (str)  — highest degree + institution detected
    raw_skills_text   (str)  — raw text of the skills section (for BM25)

Strategy:
    - Uses regex and a curated keyword dictionary (no LLM required).
    - Heuristic name detection: first non-empty line that looks like a proper name.
    - Experience years: sum of date ranges found anywhere in the text.
    - Skills: intersection of text with a curated skills vocabulary.

Usage:
    from utils.metadata import extract_metadata

    meta = extract_metadata(text)
"""

from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# Curated skills vocabulary (technical + popular frameworks + soft skills)
# ---------------------------------------------------------------------------

SKILLS_VOCABULARY: list[str] = [
    # Programming languages
    "Python", "JavaScript", "TypeScript", "Java", "C", "C++", "C#", "Go", "Rust",
    "Ruby", "PHP", "Swift", "Kotlin", "Scala", "R", "MATLAB", "Perl", "Bash",
    "Shell", "PowerShell", "SQL", "PL/SQL", "T-SQL", "Dart", "Elixir", "Haskell",
    # Web frameworks & libraries
    "Django", "Flask", "FastAPI", "Spring", "Spring Boot", "Express", "Node.js",
    "React", "Vue.js", "Angular", "Next.js", "Nuxt.js", "Svelte", "jQuery",
    "Bootstrap", "Tailwind", "Laravel", "Rails", "Ruby on Rails", "ASP.NET",
    ".NET", "Blazor", "GraphQL", "REST", "REST API", "gRPC",
    # Data & ML
    "Machine Learning", "Deep Learning", "NLP", "Natural Language Processing",
    "Computer Vision", "Reinforcement Learning", "TensorFlow", "PyTorch", "Keras",
    "scikit-learn", "XGBoost", "LightGBM", "Pandas", "NumPy", "Matplotlib",
    "Seaborn", "Plotly", "Hugging Face", "Transformers", "BERT", "GPT",
    "LLM", "RAG", "Vector Database", "Embeddings", "Langchain", "OpenAI",
    "Spark", "PySpark", "Hadoop", "Airflow", "MLflow", "Kubeflow", "DVC",
    "Feature Engineering", "Model Deployment", "A/B Testing", "Statistics",
    # Databases
    "MySQL", "PostgreSQL", "SQLite", "Oracle", "MongoDB", "Redis", "Cassandra",
    "DynamoDB", "Elasticsearch", "ChromaDB", "Pinecone", "Weaviate", "Neo4j",
    "Firebase", "Supabase", "CockroachDB",
    # Cloud & DevOps
    "AWS", "Azure", "GCP", "Google Cloud", "Docker", "Kubernetes", "Terraform",
    "Ansible", "Jenkins", "GitHub Actions", "GitLab CI", "CircleCI", "Helm",
    "Prometheus", "Grafana", "ELK Stack", "Nginx", "Apache", "Cloudflare",
    "Serverless", "Lambda", "EC2", "S3", "RDS", "CloudFormation", "Pulumi",
    # Tools & platforms
    "Git", "GitHub", "GitLab", "Bitbucket", "Jira", "Confluence", "Slack",
    "Linux", "Unix", "Windows", "macOS", "VS Code", "IntelliJ", "PyCharm",
    "Jupyter", "Postman", "Figma", "Tableau", "Power BI", "Looker",
    # Architecture & practices
    "Microservices", "Monolith", "Event-Driven", "CQRS", "DDD", "TDD", "BDD",
    "Agile", "Scrum", "Kanban", "CI/CD", "DevOps", "DataOps", "MLOps",
    "System Design", "API Design", "Database Design",
    # Soft skills (selected)
    "Leadership", "Communication", "Teamwork", "Problem Solving", "Mentoring",
    "Project Management", "Stakeholder Management",
]

# Pre-build lookup for case-insensitive matching
_SKILLS_LOWER: dict[str, str] = {s.lower(): s for s in SKILLS_VOCABULARY}

# ---------------------------------------------------------------------------
# Degree keywords
# ---------------------------------------------------------------------------

DEGREE_PATTERNS = [
    r"(Ph\.?D\.?|Doctor of Philosophy)",
    r"(M\.?S\.?|M\.?Sc\.?|Master of Science)",
    r"(M\.?E\.?|Master of Engineering)",
    r"(M\.?B\.?A\.?|Master of Business Administration)",
    r"(M\.?A\.?|Master of Arts)",
    r"(B\.?S\.?|B\.?Sc\.?|Bachelor of Science)",
    r"(B\.?E\.?|Bachelor of Engineering)",
    r"(B\.?Tech\.?|Bachelor of Technology)",
    r"(B\.?A\.?|Bachelor of Arts)",
    r"(Associate)",
    r"(Diploma)",
]

_DEGREE_REGEX = re.compile(
    "|".join(DEGREE_PATTERNS), re.IGNORECASE
)

# ---------------------------------------------------------------------------
# Date range pattern (e.g. "Jan 2018 – Mar 2021", "2019 - 2023", "2020–Present")
# ---------------------------------------------------------------------------

_DATE_RANGE_REGEX = re.compile(
    r"""
    (?:                                    # optional month
        (?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|
           May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|
           Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)
        \s+
    )?
    (\d{4})                                # start year
    \s*[-–—]\s*                            # dash/en-dash/em-dash
    (?:                                    # end: year or "Present"
        (?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|
           May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|
           Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)
        \s+
    )?
    (\d{4}|[Pp]resent|[Cc]urrent|[Nn]ow)  # end year
    """,
    re.VERBOSE,
)

# Current year for "Present" calculation
import datetime
_CURRENT_YEAR = datetime.datetime.now().year

# ---------------------------------------------------------------------------
# Name detection
# ---------------------------------------------------------------------------

_NAME_BLACKLIST = re.compile(
    r"(?i)(resume|curriculum|vitae|cv|page|profile|linkedin|github|phone|email|"
    r"address|summary|objective|experience|education|skills|certif|project|award|"
    r"reference|contact|www\.|http|\.com|@)"
)

_PROPER_NAME_REGEX = re.compile(
    r"^([A-Za-z]+(?: [A-Za-z.]+){1,3})$"
)


def _detect_name(text: str) -> str:
    """Heuristically detect the candidate's name from the first few lines."""
    for line in text.splitlines()[:15]:
        line = line.strip()
        if not line or _NAME_BLACKLIST.search(line):
            continue
        # Must look like "First Last" or "First Middle Last"
        if _PROPER_NAME_REGEX.match(line):
            # Normalise to title case if uppercase
            if line.isupper():
                return line.title()
            return line
    return "Unknown"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_metadata(text: str) -> dict[str, Any]:
    """Extract structured metadata from raw resume text.

    Args:
        text: Full raw text of a resume.

    Returns:
        dict with keys:
            name              (str)   — candidate's full name
            skills            (list)  — detected skills (canonical casing)
            experience_years  (int)   — estimated total years of experience
            education         (str)   — highest detected degree + institution
            raw_skills_text   (str)   — joined skills string (for keyword index)
    """
    name = _detect_name(text)
    skills = _extract_skills(text)
    experience_years = _estimate_experience_years(text)
    education = _extract_education(text)

    return {
        "name": name,
        "skills": skills,
        "experience_years": experience_years,
        "education": education,
        "raw_skills_text": " ".join(skills),
    }


# ---------------------------------------------------------------------------
# Internal extractors
# ---------------------------------------------------------------------------


def _extract_skills(text: str) -> list[str]:
    """Find all skills from the vocabulary present in the text."""
    text_lower = text.lower()
    found: list[str] = []
    seen: set[str] = set()

    for skill_lower, skill_canonical in _SKILLS_LOWER.items():
        # Use word-boundary matching to avoid partial matches
        pattern = r"(?<![a-z\-])" + re.escape(skill_lower) + r"(?![a-z\-])"
        if re.search(pattern, text_lower) and skill_canonical not in seen:
            found.append(skill_canonical)
            seen.add(skill_canonical)

    return sorted(found)


def _estimate_experience_years(text: str) -> int:
    """Estimate total years of experience from date ranges in the text."""
    total_months = 0
    seen_ranges: set[tuple[int, int]] = set()

    for match in _DATE_RANGE_REGEX.finditer(text):
        start_year_str, end_year_str = match.group(1), match.group(2)
        if not start_year_str:
            continue

        start_year = int(start_year_str)
        if end_year_str.lower() in ("present", "current", "now"):
            end_year = _CURRENT_YEAR
        else:
            end_year = int(end_year_str)

        # Sanity check: reasonable year range
        if not (1970 <= start_year <= _CURRENT_YEAR):
            continue
        if not (1970 <= end_year <= _CURRENT_YEAR + 1):
            continue
        if start_year > end_year:
            continue

        range_key = (start_year, end_year)
        if range_key not in seen_ranges:
            seen_ranges.add(range_key)
            total_months += (end_year - start_year) * 12

    # Convert months to years, clamp between 0 and 50
    years = min(max(total_months // 12, 0), 50)
    return years


def _extract_education(text: str) -> str:
    """Extract the highest educational qualification mentioned."""
    # Degree priority order (highest to lowest)
    degree_priority = [
        "ph.d", "phd", "doctor",
        "m.s", "msc", "m.sc", "master of science",
        "m.e", "master of engineering",
        "m.b.a", "mba", "master of business",
        "m.a", "master of arts",
        "b.s", "bsc", "b.sc", "bachelor of science",
        "b.e", "bachelor of engineering",
        "b.tech", "bachelor of technology",
        "b.a", "bachelor of arts",
        "associate",
        "diploma",
    ]

    match = _DEGREE_REGEX.search(text)
    if not match:
        return "Not specified"

    # Try to find the line containing the degree for context
    matched_degree = match.group(0)
    for line in text.splitlines():
        if matched_degree.lower() in line.lower() and len(line.strip()) > 5:
            return line.strip()[:120]  # Cap at 120 chars

    return matched_degree
