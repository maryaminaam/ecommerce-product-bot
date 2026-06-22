"""
Product Researcher agent: retrieves product information from the internal
catalog (RAG) and/or the live web depending on the query type.
Wrapped as an AgentTool so the Orchestrator can delegate to it.
"""

from google.adk.agents import Agent
from google.adk.tools import AgentTool
from src.tools.catalog_search import catalog_search_tool
from src.tools.web_search import web_search_tool

researcher_agent = Agent(
    name="product_researcher",
    model="gemini-2.5-flash",
    instruction="""
    You are a product research specialist for an e-commerce platform.
    Your job is to retrieve accurate and relevant product information.

    TOOL SELECTION RULES:
    - Use catalog_search for: product specs, technical features, dimensions,
      compatibility, warranty info, and FAQ answers from the product manual.
    - Use web_search for: current retail price, live stock availability,
      recent customer reviews (last 6 months), competitor comparisons,
      and any information that changes frequently.
    - Use BOTH tools when the query needs specs AND current pricing/reviews.

    OUTPUT FORMAT:
    Always return a structured response with:
    - product_name: the product(s) you researched
    - specs: key technical specifications (from catalog if available)
    - price_range: current price from web search
    - reviews_summary: summary of recent reviews
    - source_tags: list of sources used (knowledge_base, web, or both)
    - raw_data: the full retrieved content for the reviewer agent

    Never guess or hallucinate product specs. Only report what the tools return.
    """,
    tools=[catalog_search_tool, web_search_tool],
)

# Wrap as AgentTool so the Orchestrator can call it like a function
researcher_tool = AgentTool(agent=researcher_agent)
