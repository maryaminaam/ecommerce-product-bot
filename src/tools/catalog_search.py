"""
Catalog search tool: wraps the RAG retriever as a Google ADK FunctionTool
so agents can search the product knowledge base by natural language query.
"""

from google.adk.tools import FunctionTool
from src.rag.retriever import get_retriever


def catalog_search(query: str) -> dict:
    """
    Searches the internal product catalog using vector similarity.
    Use this for product specs, features, compatibility, and FAQs.

    Args:
        query: Natural language question about a product.

    Returns:
        Dict with 'chunks' (list of relevant text passages) and
        'source' indicating this came from the knowledge base.
    """
    retriever = get_retriever()
    results = retriever.retrieve(query)

    chunks = [
        {
            "text": r["text"],
            "source_file": r["source"],
            "relevance_score": r["score"],
        }
        for r in results
    ]

    return {
        "chunks": chunks,
        "source": "knowledge_base",
        "num_results": len(chunks),
    }


# Wrap as ADK FunctionTool so agents can call it
catalog_search_tool = FunctionTool(func=catalog_search)
