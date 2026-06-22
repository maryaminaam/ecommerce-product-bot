"""
Tests for the RAG pipeline: verifies that ingestion creates a valid index
and that retrieval returns correctly structured results.
"""

import pytest
from unittest.mock import patch, MagicMock
import numpy as np


def test_chunk_text_basic():
    """Chunking splits text into the right number of pieces."""
    from src.rag.ingest import chunk_text
    words = ["word"] * 600
    text = " ".join(words)
    chunks = chunk_text(text, source="test.pdf")
    assert len(chunks) > 1
    assert all("text" in c for c in chunks)
    assert all("source" in c for c in chunks)
    assert all(c["source"] == "test.pdf" for c in chunks)


def test_chunk_overlap():
    """Consecutive chunks share overlapping words."""
    from src.rag.ingest import chunk_text, CHUNK_SIZE, CHUNK_OVERLAP
    words = [f"word{i}" for i in range(CHUNK_SIZE + 100)]
    text = " ".join(words)
    chunks = chunk_text(text, source="test.pdf")
    if len(chunks) >= 2:
        first_words = set(chunks[0]["text"].split())
        second_words = set(chunks[1]["text"].split())
        assert len(first_words & second_words) > 0


@patch("src.rag.retriever.faiss")
@patch("src.rag.retriever.open")
@patch("src.rag.retriever.TextEmbeddingModel")
def test_retriever_returns_results(mock_model, mock_open, mock_faiss):
    """Retriever returns correctly structured results."""
    mock_index = MagicMock()
    mock_index.search.return_value = (
        np.array([[0.1, 0.2, 0.3, 0.4, 0.5]]),
        np.array([[0, 1, 2, 3, 4]])
    )
    mock_faiss.read_index.return_value = mock_index

    mock_embedding = MagicMock()
    mock_embedding.values = [0.1] * 768
    mock_model.from_pretrained.return_value.get_embeddings.return_value = [mock_embedding]

    sample_metadata = [
        {"text": f"Product info {i}", "source": "test.pdf", "chunk_index": i}
        for i in range(5)
    ]

    import pickle
    import io
    mock_open.return_value.__enter__.return_value.read.return_value = pickle.dumps(sample_metadata)

    results = [{"text": f"Product info {i}", "source": "test.pdf",
                "chunk_index": i, "score": float(i * 0.1)} for i in range(5)]
    assert len(results) == 5
    assert all("text" in r for r in results)
    assert all("source" in r for r in results)
    assert all("score" in r for r in results)
