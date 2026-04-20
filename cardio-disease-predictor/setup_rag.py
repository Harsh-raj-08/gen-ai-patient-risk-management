"""
One-time RAG vector store initialization.
Run this once (or auto-run on first app launch) to populate ChromaDB
with the 35 clinical guideline chunks.
"""

import os
import chromadb
from chromadb.utils import embedding_functions
from rag.guidelines_data import GUIDELINE_CHUNKS


CHROMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")


def build_vector_store():
    """Build (or rebuild) the ChromaDB vector store from guideline chunks."""
    print(f"[RAG] Building vector store at {CHROMA_PATH} ...")

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Delete existing collection if present (idempotent rebuild)
    try:
        client.delete_collection("cardio_guidelines")
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name="cardio_guidelines",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )

    ids = [chunk["id"] for chunk in GUIDELINE_CHUNKS]
    documents = [chunk["content"] for chunk in GUIDELINE_CHUNKS]
    metadatas = [chunk["metadata"] for chunk in GUIDELINE_CHUNKS]

    collection.add(ids=ids, documents=documents, metadatas=metadatas)

    print(f"[RAG] ✓ Indexed {len(ids)} guideline chunks into ChromaDB.")
    return collection


if __name__ == "__main__":
    build_vector_store()
