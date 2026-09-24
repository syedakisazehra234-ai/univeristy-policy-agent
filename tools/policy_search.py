import json
import os

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from crewai.tools import tool


@tool("University Policy Search")
def policy_search_tool(query: str) -> str:
    """
    Search the university policy knowledge base.
    """

    return "Policy search tool is working."


# ============================================================
# Paths
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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

MAX_CONTEXT_CHARS = int(
    os.getenv("MAX_CONTEXT_CHARS", "12000")
)


# ============================================================
# Load knowledge base
# ============================================================

if not os.path.exists(FAISS_PATH):
    raise FileNotFoundError(
        f"FAISS index not found: {FAISS_PATH}"
    )

if not os.path.exists(CHUNKS_PATH):
    raise FileNotFoundError(
        f"chunks.json not found: {CHUNKS_PATH}"
    )


index = faiss.read_index(FAISS_PATH)

with open(
    CHUNKS_PATH,
    "r",
    encoding="utf-8",
) as file:
    chunks = json.load(file)


# ============================================================
# Embedding model
# ============================================================

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)


# ============================================================
# Policy Search Tool
# ============================================================

@tool("University Policy Search")
def policy_search_tool(query: str) -> str:
    """
    Search the university policy knowledge base and return
    the most relevant policy passages with source information.
    """

    if not query or not query.strip():
        return "No search query was provided."

    query = query.strip()

    # Create query embedding
    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True,
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32",
    )

    # Search FAISS
    scores, indices = index.search(
        query_embedding,
        TOP_K,
    )

    results = []

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

        text = chunk.get(
            "text",
            "",
        )

        if not text:
            continue

        source = chunk.get(
            "source",
            chunk.get(
                "source_file",
                "Unknown source",
            ),
        )

        page = chunk.get(
            "page",
            "Unknown",
        )

        results.append(
            {
                "score": score,
                "source": source,
                "page": page,
                "text": text,
            }
        )

    if not results:
        return (
            "NO_RELEVANT_POLICY_FOUND\n\n"
            "The university policy knowledge base does not "
            "contain sufficiently relevant information for "
            "this question."
        )

    # Build context
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
Similarity: {result["score"]:.4f}

Policy text:
{result["text"]}
""".strip()
        )

    final_context = "\n\n---\n\n".join(output)

    return final_context[:MAX_CONTEXT_CHARS]
