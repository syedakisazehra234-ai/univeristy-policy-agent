import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from crewai.tools import BaseTool

from config import (
    CHUNKS_PATH,
    FAISS_INDEX_PATH,
    EMBEDDING_MODEL,
    TOP_K,
    MIN_SIMILARITY,
    MAX_CONTEXT_CHARS,
)


# =========================================================
# Helper functions
# =========================================================

def load_chunks(path: Path) -> list[dict[str, Any]]:
    """
    Load chunks.json and normalize several common JSON formats
    into a list of dictionaries.
    """

    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, list):
        chunks = data

    elif isinstance(data, dict):
        possible_keys = [
            "chunks",
            "documents",
            "data",
            "items",
            "records",
        ]

        chunks = None

        for key in possible_keys:
            if key in data and isinstance(data[key], list):
                chunks = data[key]
                break

        if chunks is None:
            raise ValueError(
                "chunks.json is a dictionary, but no supported "
                "chunk list was found."
            )

    else:
        raise ValueError(
            "chunks.json must contain a list or dictionary."
        )

    normalized = []

    for index, item in enumerate(chunks):

        if isinstance(item, str):
            normalized.append(
                {
                    "text": item,
                    "metadata": {},
                }
            )
            continue

        if not isinstance(item, dict):
            continue

        text = (
            item.get("text")
            or item.get("content")
            or item.get("page_content")
            or item.get("chunk")
            or item.get("document")
            or ""
        )

        metadata = item.get("metadata", {})

        if not isinstance(metadata, dict):
            metadata = {}

        # Also preserve useful metadata if it exists directly
        # in the chunk object.
        for key in [
            "source",
            "file_name",
            "filename",
            "document",
            "page",
            "page_number",
            "section",
            "title",
            "url",
        ]:
            if key in item and key not in metadata:
                metadata[key] = item[key]

        normalized.append(
            {
                "text": str(text),
                "metadata": metadata,
            }
        )

    if not normalized:
        raise ValueError(
            "No usable chunks were found in chunks.json."
        )

    return normalized


def format_metadata(metadata: dict[str, Any]) -> str:
    """
    Convert metadata into a compact human-readable string.
    """

    if not metadata:
        return "No metadata available."

    parts = []

    preferred_keys = [
        "source",
        "file_name",
        "filename",
        "document",
        "title",
        "section",
        "page",
        "page_number",
        "url",
    ]

    used = set()

    for key in preferred_keys:
        if key in metadata:
            value = metadata[key]

            if value is not None and str(value).strip():
                parts.append(f"{key}: {value}")
                used.add(key)

    # Include any remaining metadata.
    for key, value in metadata.items():
        if key in used:
            continue

        if value is not None and str(value).strip():
            parts.append(f"{key}: {value}")

    return " | ".join(parts)


# =========================================================
# Retriever
# =========================================================

class UniversityPolicyRetriever:
    """
    Loads the existing FAISS index and performs semantic search
    using the same embedding model used during indexing.
    """

    def __init__(self):
        if not FAISS_INDEX_PATH.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {FAISS_INDEX_PATH}"
            )

        if not CHUNKS_PATH.exists():
            raise FileNotFoundError(
                f"chunks.json not found: {CHUNKS_PATH}"
            )

        # Load FAISS index.
        self.index = faiss.read_index(
            str(FAISS_INDEX_PATH)
        )

        # Load original text chunks.
        self.chunks = load_chunks(CHUNKS_PATH)

        # Important integrity check.
        if self.index.ntotal != len(self.chunks):
            raise ValueError(
                "FAISS index and chunks.json are out of sync.\n"
                f"FAISS vectors: {self.index.ntotal}\n"
                f"JSON chunks: {len(self.chunks)}\n\n"
                "The FAISS index and chunks.json must come "
                "from the same indexing process."
            )

        # Load embedding model.
        self.embedding_model = SentenceTransformer(
            EMBEDDING_MODEL
        )

        # Check vector dimensions.
        embedding_dimension = (
            self.embedding_model.get_sentence_embedding_dimension()
        )

        if embedding_dimension != self.index.d:
            raise ValueError(
                "Embedding dimension mismatch.\n"
                f"Embedding model dimension: {embedding_dimension}\n"
                f"FAISS index dimension: {self.index.d}\n\n"
                "Make sure EMBEDDING_MODEL is exactly the same "
                "model used to create the FAISS index."
            )

    def search(
        self,
        query: str,
        top_k: int = TOP_K,
    ) -> list[dict[str, Any]]:
        """
        Search the FAISS index and return the most relevant chunks.
        """

        query = query.strip()

        if not query:
            return []

        # Create query embedding.
        query_embedding = self.embedding_model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32",
        )

        # Search FAISS.
        scores, indices = self.index.search(
            query_embedding,
            top_k,
        )

        results = []

        for score, index in zip(scores[0], indices[0]):

            if index < 0:
                continue

            if index >= len(self.chunks):
                continue

            score = float(score)

            if score < MIN_SIMILARITY:
                continue

            chunk = self.chunks[index]

            results.append(
                {
                    "text": chunk["text"],
                    "metadata": chunk["metadata"],
                    "score": score,
                }
            )

        return results

    def build_context(
        self,
        query: str,
        top_k: int = TOP_K,
    ) -> str:
        """
        Build the context that will be given to the CrewAI agent.
        """

        results = self.search(
            query=query,
            top_k=top_k,
        )

        if not results:
            return "NO_RELEVANT_POLICY_FOUND"

        context_parts = []

        current_length = 0

        for number, result in enumerate(results, start=1):

            text = result["text"].strip()

            if not text:
                continue

            metadata = format_metadata(
                result["metadata"]
            )

            score = result["score"]

            block = (
                f"[SOURCE {number}]\n"
                f"Similarity: {score:.4f}\n"
                f"Metadata: {metadata}\n"
                f"Content:\n{text}\n"
            )

            if (
                current_length + len(block)
                > MAX_CONTEXT_CHARS
            ):
                break

            context_parts.append(block)
            current_length += len(block)

        if not context_parts:
            return "NO_RELEVANT_POLICY_FOUND"

        return "\n---\n".join(context_parts)


# =========================================================
# CrewAI Tool
# =========================================================

class UniversityPolicySearchInput(BaseModel):
    query: str = Field(
        ...,
        description=(
            "The university policy question that should "
            "be searched in the policy knowledge base."
        ),
    )


class UniversityPolicySearchTool(BaseTool):
    name: str = "university_policy_search"

    description: str = (
        "Search the university's official policy knowledge "
        "base using semantic search. Use this tool whenever "
        "the user asks about university rules, policies, "
        "requirements, procedures, deadlines, attendance, "
        "examinations, admissions, fees, conduct, academic "
        "regulations, or other institutional policies."
    )

    args_schema: type[BaseModel] = UniversityPolicySearchInput

    def _run(self, query: str) -> str:
        try:
            retriever = get_retriever()

            return retriever.build_context(
                query=query,
                top_k=TOP_K,
            )

        except Exception as exc:
            return (
                "POLICY_SEARCH_ERROR: "
                f"{type(exc).__name__}: {exc}"
            )


# =========================================================
# Singleton retriever
# =========================================================

_retriever = None


def get_retriever() -> UniversityPolicyRetriever:
    """
    Create the retriever once and reuse it.
    """

    global _retriever

    if _retriever is None:
        _retriever = UniversityPolicyRetriever()

    return _retriever


# Tool instance used by the CrewAI agent.
university_policy_search = UniversityPolicySearchTool()
