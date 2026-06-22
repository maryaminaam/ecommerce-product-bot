"""
RAG retriever: loads the FAISS index built by ingest.py and retrieves
the top-k most relevant chunks for a given query using vector similarity.
"""

import os
import pickle
from pathlib import Path

import faiss
import numpy as np
from vertexai.language_models import TextEmbeddingModel
from google.cloud import aiplatform
from dotenv import load_dotenv

load_dotenv()

KNOWLEDGE_BASE_DIR = Path("knowledge_base")
INDEX_PATH = KNOWLEDGE_BASE_DIR / "index.faiss"
METADATA_PATH = KNOWLEDGE_BASE_DIR / "metadata.pkl"
TOP_K = 5


def init_vertex() -> None:
    """Initializes Vertex AI with credentials from environment."""
    aiplatform.init(
        project=os.getenv("GOOGLE_CLOUD_PROJECT"),
        location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
    )


class ProductRetriever:
    """
    Loads the FAISS index and metadata once, then serves retrieval
    requests efficiently without reloading on every query.
    """

    def __init__(self) -> None:
        """Loads FAISS index and chunk metadata from disk."""
        init_vertex()
        if not INDEX_PATH.exists():
            raise FileNotFoundError(
                "FAISS index not found. Run src/rag/ingest.py first."
            )
        self.index = faiss.read_index(str(INDEX_PATH))
        with open(METADATA_PATH, "rb") as f:
            self.metadata = pickle.load(f)
        self.model = TextEmbeddingModel.from_pretrained("text-embedding-005")

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embeds a single query string using Vertex AI.

        Args:
            query: The search query to embed.

        Returns:
            Numpy array of shape (1, embedding_dim).
        """
        embeddings = self.model.get_embeddings([query])
        return np.array([embeddings[0].values], dtype="float32")

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[dict]:
        """
        Retrieves the most relevant chunks for a query.

        Args:
            query: Natural language search query.
            top_k: Number of chunks to return.

        Returns:
            List of dicts with 'text', 'source', 'chunk_index', 'score'.
        """
        query_embedding = self.embed_query(query)
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.metadata[idx].copy()
            chunk["score"] = float(dist)
            results.append(chunk)

        return results


# Singleton instance to avoid reloading the index on every tool call
_retriever_instance: ProductRetriever | None = None


def get_retriever() -> ProductRetriever:
    """Returns a singleton ProductRetriever instance."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = ProductRetriever()
    return _retriever_instance
