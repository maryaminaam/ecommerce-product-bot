"""
Orchestrator agent: the central coordinator of the multi-agent system.
Classifies user intent, routes to researcher and reviewer agents,
implements the feedback loop for low-confidence results, and manages
session state across the conversation.
"""

import json
from google.adk.agents import Agent
from google.adk.tools import AgentTool
from src.agents.researcher import researcher_tool
from src.agents.reviewer import reviewer_tool
from src.observability.logger import log_agent_call, log_pipeline_start, Timer


ORCHESTRATOR_INSTRUCTION = """
You are the coordinator of an e-commerce product intelligence system.
Your job is to understand the user's intent and orchestrate the researcher
and reviewer agents to produce the best possible product recommendation.

STEP 1 — CLASSIFY INTENT:
Classify every incoming query as exactly one of:
- product_lookup: user wants specs or info on ONE specific product
- comparison: user wants to compare TWO OR MORE specific products
- recommendation: user wants help choosing ("best X for Y", "which should I buy")

STEP 2 — EXECUTE THE RIGHT PLAN:

For product_lookup:
  1. Call researcher_tool with the product name and query
  2. Return the specs and info directly — no reviewer needed

For comparison:
  1. Call researcher_tool once for each product (mention both in one call
     and instruct researcher to retrieve data for each)
  2. Pass ALL retrieved data to reviewer_tool with the original query
  3. Return the reviewer's structured recommendation

For recommendation:
  1. Call researcher_tool with a broad query covering likely candidates
  2. Pass results to reviewer_tool
  3. If reviewer returns needs_retry = true, call researcher_tool again
     with a more specific refined query, then call reviewer_tool again
  4. Maximum 2 retries — if still low confidence, return best available answer

STEP 3 — FEEDBACK LOOP:
After getting the reviewer's response, check the "needs_retry" field.
If true and retries_remaining > 0:
  - Refine the search query based on retry_reason
  - Call researcher_tool again with the refined query
  - Call reviewer_tool again with the new data
  - Decrement retries_remaining

STEP 4 — FINAL RESPONSE FORMAT:
Always end with a clean, friendly summary that includes:
- The top recommendation and why
- The runner-up option
- The key trade-off the user should know about
- Current approximate price for each option
- A confidence note if confidence was low

Keep responses conversational and helpful. The user is a shopper, not a tech expert.
"""


def create_orchestrator() -> Agent:
    """
    Creates and returns the configured Orchestrator agent with all
    sub-agent tools attached.

    Returns:
        Configured Google ADK Agent instance.
    """
    orchestrator = Agent(
        name="orchestrator",
        model="gemini-2.5-flash",
        instruction=ORCHESTRATOR_INSTRUCTION,
        tools=[researcher_tool, reviewer_tool],
    )
    return orchestrator


def run_query(query: str, session_context: dict | None = None) -> str:
    """
    Runs a single query through the full multi-agent pipeline.
    This is the main entry point called by main.py.

    Args:
        query: The user's natural language product question.
        session_context: Optional dict with prior session state.

    Returns:
        The orchestrator's final response as a string.
    """
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService

    session_service = InMemorySessionService()
    session = session_service.create_session(
        app_name="ecommerce_bot",
        user_id="demo_user",
    )

    orchestrator = create_orchestrator()

    runner = Runner(
        agent=orchestrator,
        app_name="ecommerce_bot",
        session_service=session_service,
    )

    with Timer() as t:
        response = runner.run(
            user_id="demo_user",
            session_id=session.id,
            new_message=query,
        )

    log_agent_call(
        agent_name="orchestrator",
        query=query,
        tools_called=["researcher_tool", "reviewer_tool"],
        latency_ms=t.elapsed_ms,
        output_summary=str(response)[:200],
    )

    return str(response)
