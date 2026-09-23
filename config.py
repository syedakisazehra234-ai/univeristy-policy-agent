import os
from pathlib import Path

from dotenv import load_dotenv


# Load local .env when running locally.
# Streamlit Cloud secrets are also available through os.environ.
load_dotenv()


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"

FAISS_INDEX_PATH = KNOWLEDGE_BASE_DIR / "faiss.index"
CHUNKS_PATH = KNOWLEDGE_BASE_DIR / "chunks.json"


# ---------------------------------------------------------
# API configuration
# ---------------------------------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b",
).strip()


# ---------------------------------------------------------
# Embedding configuration
# ---------------------------------------------------------

# IMPORTANT:
# This MUST be the same embedding model that was used when
# faiss.index was originally created.
#
# Change this value if your original indexing notebook used
# a different model.

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
).strip()


# ---------------------------------------------------------
# Retrieval configuration
# ---------------------------------------------------------

TOP_K = int(os.getenv("TOP_K", "5"))

MIN_SIMILARITY = float(
    os.getenv("MIN_SIMILARITY", "0.25")
)

MAX_CONTEXT_CHARS = int(
    os.getenv("MAX_CONTEXT_CHARS", "18000")
)


def validate_configuration():
    """
    Validate important configuration before the application starts.
    """

    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is missing. "
            "Add it to Streamlit Secrets or your local .env file."
        )

    if not FAISS_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {FAISS_INDEX_PATH}"
        )

    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"chunks.json not found: {CHUNKS_PATH}"
        )

    if TOP_K < 1:
        raise ValueError("TOP_K must be at least 1.")

    if MIN_SIMILARITY < -1 or MIN_SIMILARITY > 1:
        raise ValueError(
            "MIN_SIMILARITY must be between -1 and 1."
        )
