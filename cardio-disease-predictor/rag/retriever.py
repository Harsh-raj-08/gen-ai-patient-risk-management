"""
RAG Retriever — ChromaDB-based guideline retrieval.
Queries the vector store for relevant clinical guidelines based on risk profile topics.
"""

import os
import chromadb
from chromadb.utils import embedding_functions


# Persistent path for the ChromaDB store
CHROMA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chroma_db")

# Use the default sentence-transformers embedding function
_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)


def get_collection():
    """Return the guidelines ChromaDB collection (read-only)."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_collection(name="cardio_guidelines", embedding_function=_ef)


def retrieve_guidelines(query_text: str, top_k: int = 4) -> list[dict]:
    """
    Query ChromaDB for the most relevant guideline chunks.

    Args:
        query_text: Free-text query (typically the risk profile).
        top_k: Number of chunks to return.

    Returns:
        List of dicts with keys: topic, source, content.
    """
    try:
        collection = get_collection()
        results = collection.query(
            query_texts=[query_text],
            n_results=top_k,
            include=["documents", "metadatas"]
        )

        guidelines = []
        if results and results["documents"]:
            for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                guidelines.append({
                    "topic": meta.get("topic", "unknown"),
                    "source": meta.get("source", "unknown"),
                    "content": doc
                })
        return guidelines

    except Exception as e:
        print(f"[RAG] Retrieval error: {e}")
        return [{
            "topic": "fallback",
            "source": "WHO CVD Guidelines 2023",
            "content": (
                "General cardiovascular prevention: maintain blood pressure below 140/90 mmHg, "
                "engage in 150+ minutes of moderate exercise per week, follow a heart-healthy diet "
                "low in sodium and saturated fat, avoid tobacco, and limit alcohol consumption."
            )
        }]
