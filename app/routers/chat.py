"""
app/routers/chat.py

Endpoint:
  POST /api/chat   ← HTML: sendMessage() → callClaudeAPI() replaced by this

This is where the Anthropic/Groq API call moves FROM the browser TO the server.
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.core.config import Settings, get_settings
from app.core.database import (
    create_session, get_session, add_message,
    update_session_title, get_messages,
)
from app.services.rag_pipeline import run_rag
from app.services.vector_store import VectorStore

router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatRequest(BaseModel):
    session_id: str
    message: str = Field(min_length=1, max_length=4000)


def _vs():
    from app.dependencies import get_vector_store
    return get_vector_store()


@router.post("")
async def chat(
    req: ChatRequest,
    settings: Settings = Depends(get_settings),
    vector_store: VectorStore = Depends(_vs),
):
    """
    WHAT THIS DOES — this replaces callClaudeAPI() in the HTML:

    1. Check documents exist → 400 if none indexed
       (HTML currently checks documents.length > 0 in JS)

    2. Ensure session exists (create if missing)
       (HTML manages sessions in JS; backend needs them in DB)

    3. Save user message to DB

    4. Run the full LangGraph RAG pipeline:
         retrieve → grade → [rewrite →]* generate
       This is the core: FAISS similarity search + Groq LLM

    5. Save assistant answer + sources to DB

    6. Auto-title session from first message (so History shows readable titles)

    7. Return {answer, sources, rewritten_query, chunk_count, latency_ms}

    RESPONSE → HTML:
      answer           → appendMessage('ai', answer, source)
      sources[0]       → msg-source chip below the bubble
      rewritten_query  → console.log (debug info)
      chunk_count      → could show in UI later

    WHY move this to the backend?
    - API key security: never expose GROQ_API_KEY to the browser
    - RAG requires server-side FAISS vector search
    - Session persistence across page reloads
    """
    # 1. Must have documents
    if not vector_store.has_documents():
        raise HTTPException(
            400,
            "No documents are indexed. Please upload documents first."
        )

    # 2. Ensure session exists
    if not get_session(req.session_id):
        create_session(req.session_id)

    # 3. Save user message
    add_message(req.session_id, "user", req.message)

    # 4. Run RAG pipeline
    try:
        result = await run_rag(req.message, vector_store, settings)
    except Exception as e:
        raise HTTPException(500, f"RAG pipeline error: {e}")

    # 5. Save answer
    add_message(
        req.session_id,
        "assistant",
        result["answer"],
        sources=result["sources"],
    )

    # 6. Auto-title from first user message
    msgs = get_messages(req.session_id)
    if len(msgs) <= 2:
        update_session_title(req.session_id, req.message[:80])

    return result
