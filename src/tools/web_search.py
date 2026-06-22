"""
Web search tool: wraps the Tavily API as a Google ADK FunctionTool
so agents can fetch live product prices, reviews, and availability.
"""

import os
from tavily import TavilyClient
from google.adk.tools import FunctionTool
from dotenv import load_dotenv

load_dotenv()


def web_search(query: str) -> dict:
    """
    Searches the web for current product prices, reviews, and availability.
    Use this for real-time information not in the internal catalog, such as
    competitor pricing, recent user reviews, and stock availability.

    Args:
        query: Natural language search query about a product.

    Returns:
        Dict with 'results' (list of web results) and 'source' = 'web'.
    """
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    response = client.search(
        query=query,
        search_depth="advanced",
        max_results=5,
        include_answer=True,
    )

    results = [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("content", ""),
            "published_date": r.get("published_date", "unknown"),
            "score": r.get("score", 0.0),
        }
        for r in response.get("results", [])
    ]

    return {
        "results": results,
        "source": "web",
        "answer_summary": response.get("answer", ""),
        "num_results": len(results),
    }


# Wrap as ADK FunctionTool so agents can call it
web_search_tool = FunctionTool(func=web_search)
