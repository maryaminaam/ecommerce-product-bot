"""
Observability module for logging agent activity, latency, and tool usage.
Writes structured JSON logs to both console and logs/agent_activity.log.
"""

import logging
import json
import time
import os
from datetime import datetime

os.makedirs("logs", exist_ok=True)

logger = logging.getLogger("ecommerce_bot")
logger.setLevel(logging.INFO)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# File handler
file_handler = logging.FileHandler("logs/agent_activity.log")
file_handler.setLevel(logging.INFO)

# JSON formatter
class JsonFormatter(logging.Formatter):
    """Formats log records as JSON for Google Cloud Logging compatibility."""
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "severity": record.levelname,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra"):
            log_entry.update(record.extra)
        return json.dumps(log_entry)

formatter = JsonFormatter()
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)


def log_agent_call(
    agent_name: str,
    query: str,
    tools_called: list[str],
    latency_ms: float,
    output_summary: str,
    token_count: int = 0,
) -> None:
    """
    Logs a single agent invocation with all relevant metadata.

    Args:
        agent_name: Name of the agent being invoked.
        query: The input query the agent received.
        tools_called: List of tool names the agent used.
        latency_ms: How long the agent took in milliseconds.
        output_summary: Short summary of the agent's output.
        token_count: Approximate token count if available.
    """
    extra = {
        "agent_name": agent_name,
        "query": query,
        "tools_called": tools_called,
        "latency_ms": round(latency_ms, 2),
        "output_summary": output_summary,
        "token_count": token_count,
    }
    logger.info("Agent invocation", extra={"extra": extra})


def log_pipeline_start(query: str, intent: str) -> None:
    """Logs the start of a full pipeline run with detected intent."""
    extra = {"query": query, "detected_intent": intent}
    logger.info("Pipeline started", extra={"extra": extra})


def log_pipeline_end(query: str, total_latency_ms: float) -> None:
    """Logs the completion of a full pipeline run."""
    extra = {"query": query, "total_latency_ms": round(total_latency_ms, 2)}
    logger.info("Pipeline completed", extra={"extra": extra})


class Timer:
    """Context manager for measuring latency in milliseconds."""

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.perf_counter() - self.start) * 1000
