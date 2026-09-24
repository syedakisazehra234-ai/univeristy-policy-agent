import json
import os

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from crewai.tools import tool


# ============================================================
# Paths
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

FAISS_PATH = os.path.join(
    BASE_DIR,
    "faiss_index",
    "faiss.index",
)

CHUNKS_PATH = os.path.join(
    BASE_DIR,
    "chunks.json",
)


# ============================================================
# Configuration
# ============================================================

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "all-MiniLM-L6-v2",
)

TOP_K = int(
    os.getenv("TOP_K", "5")
)

MIN_SIMILARITY = float(
    os.getenv("MIN_SIMILARITY", "0.25")
)


# ============================================================
# Load FAISS index
# ============================================================

if not os.path.exists(FAISS_PATH):
    raise FileNotFoundError(
        f"FAISS index not found: {FAISS_PATH}"
    )

index = faiss.read_index(FAISS_PATH)


# ============================================================
# Load chunks
# ============================================================

if not os.path.exists(CHUNKS_PATH):
    raise FileNotFoundError(
        f"chunks.json not found: {CHUNKS_PATH}"
    )

with open(
    CHUNKS_PATH,
    "r",
    encoding="utf-8",
) as file:
    chunks = json.load(file)


# ============================================================
# Load embedding model
# ============================================================

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)


# ============================================================
# University Policy Search Tool
# ============================================================

@tool("University Policy Search")
def policy_search_tool(query: str) -> str:
    """
    Search the university policy knowledge base using
    semantic similarity.
    """

    if not query or not query.strip():
        return "No search query was provided."

    query = query.strip()

    # --------------------------------------------------------
    # Create embedding for user's question
    # --------------------------------------------------------

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True,
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32",
    )

    # --------------------------------------------------------
    # Search FAISS
    # --------------------------------------------------------

    scores, indices = index.search(
        query_embedding,
        TOP_K,
    )

    results = []

    # --------------------------------------------------------
    # Get matching chunks
    # --------------------------------------------------------

    for score, idx in zip(
        scores[0],
        indices[0],
    ):

        if idx < 0 or idx >= len(chunks):
            continue

        score = float(score)

        if score < MIN_SIMILARITY:
            continue

        chunk = chunks[idx]

        text = chunk.get("text", "")

        if not text:
            continue

        source = (
            chunk.get("source")
            or chunk.get("source_file")
            or chunk.get("filename")
            or "Unknown source"
        )

        page = chunk.get(
            "page",
            "Unknown page",
        )

        chunk_number = chunk.get(
            "chunk_number",
            "Unknown",
        )

        results.append(
            {
                "score": score,
                "source": source,
                "page": page,
                "chunk": chunk_number,
                "text": text,
            }
        )

    # --------------------------------------------------------
    # No relevant results
    # --------------------------------------------------------

    if not results:

        return (
            "NO_RELEVANT_POLICY_FOUND\n\n"
            "The university policy knowledge base does not "
            "contain sufficiently relevant information to "
            "answer this question."
        )

    # --------------------------------------------------------
    # Format results
    # --------------------------------------------------------

    output = []

    for number, result in enumerate(
        results,
        start=1,
    ):

        output.append(
            f"""
SOURCE {number}

Source: {result["source"]}
Page: {result["page"]}
Chunk: {result["chunk"]}
Similarity: {result["score"]:.4f}

Policy text:
{result["text"]}
""".strip()
        )

    return "\n\n--------------------\n\n".join(output)
