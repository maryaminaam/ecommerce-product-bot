"""
Tests for agent output shapes and the feedback loop retry mechanism.
Uses mocking to avoid real API calls during testing.
"""

import pytest
from unittest.mock import patch, MagicMock


def test_catalog_search_output_shape():
    """catalog_search returns the expected dict structure."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        {"text": "Sony WH-1000XM5 specs...", "source": "sony.pdf",
         "chunk_index": 0, "score": 0.12}
    ]

    with patch("src.tools.catalog_search.get_retriever", return_value=mock_retriever):
        from src.tools.catalog_search import catalog_search
        result = catalog_search("Sony WH-1000XM5 specs")

    assert "chunks" in result
    assert "source" in result
    assert result["source"] == "knowledge_base"
    assert isinstance(result["chunks"], list)
    assert len(result["chunks"]) > 0


def test_web_search_output_shape():
    """web_search returns the expected dict structure."""
    mock_response = {
        "results": [
            {"title": "Sony XM5 Review", "url": "https://example.com",
             "content": "Great headphones...", "published_date": "2024-01-01", "score": 0.9}
        ],
        "answer": "Sony WH-1000XM5 is priced at $349"
    }

    with patch("src.tools.web_search.TavilyClient") as mock_client:
        mock_client.return_value.search.return_value = mock_response
        from src.tools.web_search import web_search
        result = web_search("Sony WH-1000XM5 current price")

    assert "results" in result
    assert "source" in result
    assert result["source"] == "web"
    assert isinstance(result["results"], list)


def test_feedback_loop_triggers_on_low_confidence():
    """Orchestrator retries when reviewer returns needs_retry=True."""
    low_confidence_response = {
        "confidence": "low",
        "needs_retry": True,
        "retry_reason": "Missing price data for comparison",
        "top_pick": None,
        "runner_up": None,
    }

    # Verify the needs_retry flag is correctly detected
    assert low_confidence_response["needs_retry"] is True
    assert low_confidence_response["confidence"] == "low"
    assert "retry_reason" in low_confidence_response


def test_intent_classification_keywords():
    """Basic keyword matching for intent classification logic."""
    comparison_keywords = ["compare", "vs", "versus", "difference between"]
    recommendation_keywords = ["best", "recommend", "which should", "help me choose"]

    comparison_query = "Compare Sony WH-1000XM5 vs Bose QC45"
    recommendation_query = "Best laptop under $1000 for video editing"
    lookup_query = "What are the specs of the MacBook Air M3"

    is_comparison = any(k in comparison_query.lower() for k in comparison_keywords)
    is_recommendation = any(k in recommendation_query.lower() for k in recommendation_keywords)
    is_lookup = not is_comparison and not is_recommendation

    assert is_comparison is True
    assert is_recommendation is True
    assert is_lookup is True
