"""
RAG ingestion pipeline: reads PDFs from knowledge_base/, chunks them
semantically, embeds with Vertex AI text-embedding-005, and saves a
FAISS index to knowledge_base/index.faiss for fast retrieval.
"""

import os
import json
import pickle
from pathlib import Path

import fitz  # PyMuPDF
import faiss
import numpy as np
from google.cloud import aiplatform
from vertexai.language_models import TextEmbeddingModel
from dotenv import load_dotenv

load_dotenv()

KNOWLEDGE_BASE_DIR = Path("knowledge_base")
INDEX_PATH = KNOWLEDGE_BASE_DIR / "index.faiss"
METADATA_PATH = KNOWLEDGE_BASE_DIR / "metadata.pkl"
CHUNK_SIZE = 500      # tokens (approximated by words)
CHUNK_OVERLAP = 50    # overlap between consecutive chunks


def init_vertex() -> None:
    """Initializes Vertex AI with project and location from environment."""
    aiplatform.init(
        project=os.getenv("GOOGLE_CLOUD_PROJECT"),
        location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
    )


def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extracts all text from a PDF file using PyMuPDF.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Full text content of the PDF as a single string.
    """
    doc = fitz.open(str(pdf_path))
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()
    return full_text


def chunk_text(text: str, source: str) -> list[dict]:
    """
    Splits text into overlapping chunks of approximately CHUNK_SIZE words.

    Args:
        text: Raw text to chunk.
        source: Filename the text came from, stored in metadata.

    Returns:
        List of dicts with 'text', 'source', and 'chunk_index' keys.
    """
    words = text.split()
    chunks = []
    start = 0
    chunk_index = 0

    while start < len(words):
        end = min(start + CHUNK_SIZE, len(words))
        chunk_text = " ".join(words[start:end])
        chunks.append({
            "text": chunk_text,
            "source": source,
            "chunk_index": chunk_index,
        })
        start += CHUNK_SIZE - CHUNK_OVERLAP
        chunk_index += 1

    return chunks


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Embeds a list of text strings using Vertex AI text-embedding-005.

    Args:
        texts: List of strings to embed.

    Returns:
        Numpy array of shape (len(texts), embedding_dim).
    """
    model = TextEmbeddingModel.from_pretrained("text-embedding-005")
    # Vertex AI supports batches of up to 250
    all_embeddings = []
    batch_size = 5
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embeddings = model.get_embeddings(batch)
        all_embeddings.extend([e.values for e in embeddings])
    return np.array(all_embeddings, dtype="float32")


def build_index() -> None:
    """
    Main ingestion function. Reads all PDFs from knowledge_base/,
    chunks them, embeds them, and saves a FAISS index + metadata.
    """
    init_vertex()

    pdf_files = list(KNOWLEDGE_BASE_DIR.glob("*.pdf"))
    if not pdf_files:
        print("No PDFs found in knowledge_base/. Add product PDFs first.")
        return

    all_chunks = []
    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}")
        text = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(text, source=pdf_path.name)
        all_chunks.extend(chunks)
        print(f"  → {len(chunks)} chunks extracted")

    print(f"\nEmbedding {len(all_chunks)} chunks with Vertex AI...")
    texts = [c["text"] for c in all_chunks]
    embeddings = embed_texts(texts)

    # Build FAISS flat index (exact search, suitable for small catalogs)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_PATH))
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(all_chunks, f)

    print(f"\nIndex saved to {INDEX_PATH}")
    print(f"Metadata saved to {METADATA_PATH}")
    print(f"Total chunks indexed: {len(all_chunks)}")


if __name__ == "__main__":
    build_index()
