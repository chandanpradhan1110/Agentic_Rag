"""
app/routers/health.py

Endpoints:
  GET /api/health   ← HTML: Settings panel status check (optional wiring)
  GET /api/stats    ← HTML: renderDataset() stats cards
"""
from fastapi import APIRouter, Depends
from app.core.config import Settings, get_settings
from app.core.database import get_stats

router = APIRouter(tags=["health"])


@router.get("/api/health")
async def health(settings: Settings = Depends(get_settings)):
    """
    WHAT THIS DOES:
    Returns server status. HTML Settings panel can call this to show
    whether Groq API key is configured and how many docs are indexed.
    """
    from app.dependencies import get_vector_store
    vs = get_vector_store()
    stats = get_stats()
    return {
        "status":           "ok",
        "groq_model":       settings.groq_model,
        "documents_indexed": stats["total_documents"],
        "vector_index_size": vs.total_vectors,
        "groq_configured":  bool(settings.groq_api_key),
    }


@router.get("/api/stats")
async def stats():
    """
    WHAT THIS DOES:
    HTML Dataset panel calls this to populate:
      statDocs   ← total_documents
      statChunks ← total_chunks
      statSize   ← total_size_bytes (HTML formats with formatSize())

    Also used for docCountBadge in the Chat header.
    """
    return get_stats()
