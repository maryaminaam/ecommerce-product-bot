"""
Main entry point for the E-Commerce Product Intelligence Bot.
"""
import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part
from src.agents.orchestrator import create_orchestrator

app = FastAPI(
    title="E-Commerce Product Intelligence Bot",
    description="Multi-agent shopping assistant powered by Google ADK",
    version="1.0.0",
)

session_service = InMemorySessionService()

class QueryRequest(BaseModel):
    user_id: str
    query: str

class QueryResponse(BaseModel):
    user_id: str
    query: str
    response: str

@app.get("/", response_class=HTMLResponse)
async def chat_ui():
    with open("src/ui.html") as f:
        return f.read()

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "ecommerce-bot"}

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    try:
        session = await session_service.create_session(
            app_name="ecommerce_bot",
            user_id=request.user_id,
        )
        orchestrator = create_orchestrator()
        runner = Runner(
            agent=orchestrator,
            app_name="ecommerce_bot",
            session_service=session_service,
        )
        message = Content(role="user", parts=[Part(text=request.query)])
        response_text = ""
        async for event in runner.run_async(
            user_id=request.user_id,
            session_id=session.id,
            new_message=message,
        ):
            if hasattr(event, 'is_final_response') and event.is_final_response():
                if hasattr(event, 'content') and event.content:
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text:
                            response_text += part.text
        return QueryResponse(
            user_id=request.user_id,
            query=request.query,
            response=response_text,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
