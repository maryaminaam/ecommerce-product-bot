# E-Commerce Product Intelligence Bot

A Multi-Agent AI shopping assistant built with Google ADK for a university capstone project.

## What it does
Ask questions about products and get intelligent recommendations powered by 3 AI agents working together.

## Live Demo
https://ecommerce-bot-895777686561.us-central1.run.app

## Tech Stack
- Google ADK — multi-agent orchestration
- Gemini 2.5 Flash — LLM for all agents
- Vertex AI — text embeddings
- FAISS — vector database
- Tavily API — live web search
- FastAPI — REST API
- Google Cloud Run — deployment

## Agents
- **Orchestrator** — classifies intent and routes tasks
- **Researcher** — searches knowledge base and web
- **Reviewer/QA** — scores and ranks recommendations

## Setup
1. Clone the repo
2. Copy `.env.example` to `.env` and fill in your API keys
3. Install dependencies: `pip install -r requirements.txt`
4. Build RAG index: `python src/rag/ingest.py`
5. Run: `python src/main.py`
