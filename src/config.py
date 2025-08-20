from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_PATH = PROJECT_ROOT / "data" / "docs"
DB_PATH = PROJECT_ROOT / "chroma_db"

# Models (CPU-friendly defaults)
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # embeddings
GEN_MODEL_ID = "google/flan-t5-base"                  # generator (you can use flan-t5-small too)

# Context sizing (keep small to avoid exceeding model limits on CPU)
RETRIEVER_K = 2           # how many chunks to retrieve
CONTEXT_CHARS = 300       # characters per chunk placed into the prompt
MAX_TOTAL_CONTEXT = 1200  # hard cap across all chunks (characters)
