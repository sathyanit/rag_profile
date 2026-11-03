# RAG Profile — Resume RAG System & Job Matching Engine

A complete **Retrieval-Augmented Generation (RAG)** pipeline for intelligent resume processing and job matching. Built with `sentence-transformers`, `ChromaDB`, and BM25 hybrid search — **no API keys required**.

---

## Project Structure

```
rag_profile/
├── resume_rag.py          # Part A: Ingest resumes → embed → store in ChromaDB
├── job_matcher.py         # Part B: Match job descriptions → top-K ranked JSON
├── utils/
│   ├── pdf_loader.py      # Load PDF, DOCX, TXT resume files
│   ├── chunker.py         # Section-aware chunking (Education, Experience, Skills...)
│   ├── metadata.py        # Extract Name, Skills, Experience Years, Education
│   └── scorer.py          # Hybrid scoring, filtering, reasoning
├── sample_resumes/        # 5 synthetic resumes (.txt) for demo/testing
├── resumes/               # Drop your real PDFs/DOCXs here
├── chroma_db/             # Auto-created ChromaDB persistent store
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. (Optional) Configure environment

```bash
copy .env.example .env   # Windows
# cp .env.example .env   # macOS/Linux
# Edit .env to customise model, paths, etc.
```

---

## Part A — Resume Ingestion

```bash
# Ingest the included sample resumes
python resume_rag.py --resumes-dir sample_resumes/

# Ingest your own resumes (PDF, DOCX, TXT supported)
python resume_rag.py --resumes-dir resumes/

# Full re-index (clear existing data first)
python resume_rag.py --resumes-dir sample_resumes/ --reset

# Verbose mode (shows per-chunk details)
python resume_rag.py --resumes-dir sample_resumes/ --verbose
```

**What it does:**
1. Discovers all `.pdf`, `.docx`, `.txt` files in the directory
2. Extracts text with layout-preserving parsing
3. Chunks by section (Education, Experience, Skills, etc.)
4. Extracts metadata: name, skills, experience years, education
5. Embeds each chunk with `all-MiniLM-L6-v2` (384-dim vectors)
6. Upserts into ChromaDB collection `resumes`

---

## Part B — Job Matching

```bash
# Basic job matching (returns top-10 by default)
python job_matcher.py --jd "Senior Python Machine Learning Engineer with TensorFlow and model deployment experience"

# With hard filters and custom top-k
python job_matcher.py \
  --jd "Senior ML Engineer with cloud experience" \
  --top-k 5 \
  --require "5+ years Python" \
  --require "TensorFlow"

# Multiple must-have requirements
python job_matcher.py \
  --jd "DevOps Engineer for Kubernetes platform" \
  --require "must have Kubernetes" \
  --require "AWS"

# Load JD from file, save results to JSON
python job_matcher.py \
  --jd-file job_description.txt \
  --output results.json
```

---

## Output Format

```json
{
  "job_description": "Senior Python ML Engineer...",
  "top_matches": [
    {
      "candidate_name": "Alice Chen",
      "resume_path": "sample_resumes/alice_chen.txt",
      "match_score": 92.5,
      "matched_skills": ["Python", "Machine Learning", "TensorFlow"],
      "relevant_excerpts": ["Led development of real-time recommendation engine..."],
      "reasoning": "Alice Chen is a strong match (score: 92.5/100). Matched skills: Python, TensorFlow, Machine Learning...",
      "experience_years": 5,
      "education": "Master of Science in Computer Science — Stanford University, 2019",
      "sections_matched": ["EXPERIENCE", "SKILLS", "PROJECTS"]
    }
  ],
  "total_candidates_evaluated": 5,
  "candidates_after_filter": 5,
  "model_used": "all-MiniLM-L6-v2"
}
```

---

## Architecture

```
Job Description
      │
      ▼
 SentenceTransformer            ┌─────────────────────┐
  (all-MiniLM-L6-v2)           │     ChromaDB         │
      │ embed                   │  (cosine similarity) │
      │                         │  chunk embeddings    │
      ▼                         └──────────┬──────────┘
  JD Embedding ──── query ─────────────────┘
                                           │ top K*3 chunks
                                           ▼
                               BM25 Keyword Scoring
                                           │
                                           ▼
                               Hybrid Score (70% semantic
                                          + 30% keyword)
                                           │
                                           ▼
                               Aggregate per resume
                               (60% best + 40% avg)
                                           │
                                           ▼
                               Hard Requirement Filters
                               ("5+ years Python", etc.)
                                           │
                                           ▼
                               Top-K JSON Output
```

---

## Supported Hard Filters

| Format | Example |
|---|---|
| `N+ years Skill` | `"5+ years Python"` |
| `must have Skill` | `"must have Docker"` |
| `Skill` (plain) | `"TensorFlow"` |

---

## Supported File Formats

| Format | Parser |
|---|---|
| `.pdf` | pdfplumber (layout-aware) |
| `.docx` | python-docx |
| `.txt` | built-in (UTF-8 / latin-1) |

---

## Adding Real Resumes

1. Create a `resumes/` directory in the project root.
2. Drop `.pdf` or `.docx` resume files into it.
3. Run: `python resume_rag.py --resumes-dir resumes/`
4. Run job_matcher.py as normal.

---

## Environment Variables (`.env`)

| Variable | Default | Description |
|---|---|---|
| `EMBEDDING_BACKEND` | `local` | `local` \| `openai` \| `cohere` |
| `LOCAL_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | HuggingFace model name |
| `CHROMA_DB_PATH` | `./chroma_db` | ChromaDB storage directory |
| `CHROMA_COLLECTION_NAME` | `resumes` | ChromaDB collection name |
| `DEFAULT_TOP_K` | `10` | Default number of top results |
