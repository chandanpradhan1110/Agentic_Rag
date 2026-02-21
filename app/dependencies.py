"""
app/dependencies.py

Shared application state accessed by all routers.
VectorStore is created ONCE at startup and reused for every request.
"""
from app.services.vector_store import VectorStore

_vector_store: VectorStore | None = None


def set_vector_store(vs: VectorStore):
    global _vector_store
    _vector_store = vs


def get_vector_store() -> VectorStore:
    if _vector_store is None:
        raise RuntimeError("VectorStore not initialized — check lifespan startup")
    return _vector_store
