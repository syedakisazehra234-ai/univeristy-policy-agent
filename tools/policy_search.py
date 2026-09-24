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
    os.getenv("MIN_SIMILARITY", "0.20")
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
# Load chunks.json
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
    chunks_data = json.load(file)


# ============================================================
# IMPORTANT:
# Your chunks.json has this structure:
#
# {
#     "total_chunks": 270,
#     "chunks": [...]
# }
# ============================================================

if isinstance(chunks_data, dict):

    chunks = chunks_data.get("chunks", [])

else:

    chunks = chunks_data


if not chunks:
    raise ValueError(
        "No chunks were found in chunks.json."
    )


# ============================================================
# Verify FAISS/chunks alignment
# ============================================================

if index.ntotal != len(chunks):

    raise ValueError(
        "FAISS index and chunks.json do not match. "
        f"FAISS contains {index.ntotal} vectors, "
        f"but chunks.json contains {len(chunks)} chunks."
    )


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
    semantic similarity and return relevant policy passages.
    """

    if not query or not query.strip():

        return (
            "NO_SEARCH_QUERY\n"
            "No policy search query was provided."
        )

    query = query.strip()

    # --------------------------------------------------------
    # Create embedding
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
    # Process search results
    # --------------------------------------------------------

    for score, idx in zip(
        scores[0],
        indices[0],
    ):

        idx = int(idx)
        score = float(score)

        if idx < 0 or idx >= len(chunks):
            continue

        if score < MIN_SIMILARITY:
            continue

        chunk = chunks[idx]

        text = str(
            chunk.get("text", "")
        ).strip()

        if not text:
            continue

        source_file = chunk.get(
            "source_file",
            "Unknown source",
        )

        page_number = chunk.get(
            "page_number",
            "Unknown page",
        )

        chunk_id = chunk.get(
            "chunk_id",
            f"chunk_{idx}",
        )

        results.append(
            {
                "score": score,
                "source_file": source_file,
                "page_number": page_number,
                "chunk_id": chunk_id,
                "text": text,
            }
        )

    # --------------------------------------------------------
    # No results
    # --------------------------------------------------------

    if not results:

        return (
            "NO_RELEVANT_POLICY_FOUND\n\n"
            "No sufficiently relevant university policy "
            "information was found in the knowledge base."
        )

    # --------------------------------------------------------
    # Format results for CrewAI
    # --------------------------------------------------------

    output = [
        "RELEVANT UNIVERSITY POLICY INFORMATION:",
        "",
    ]

    for number, result in enumerate(
        results,
        start=1,
    ):

        output.append(
            f"SOURCE {number}"
        )

        output.append(
            f"Document: {result['source_file']}"
        )

        output.append(
            f"Page: {result['page_number']}"
        )

        output.append(
            f"Chunk ID: {result['chunk_id']}"
        )

        output.append(
            f"Similarity: {result['score']:.4f}"
        )

        output.append(
            f"Policy text: {result['text']}"
        )

        output.append(
            "------------------------------"
        )

    return "\n".join(output)
