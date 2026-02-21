"""
main.py — FastAPI entry point for Agentic RAG

Run:
    uvicorn main:app --reload --port 8000

Then open: http://127.0.0.1:8000
"""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.core.config import get_settings
from app.core.database import init_db
from app.dependencies import set_vector_store
from app.services.vector_store import VectorStore
from app.routers import chat, documents, sessions, health

settings = get_settings()


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs ONCE on startup:
    1. Create all data directories
    2. Initialize SQLite (CREATE TABLE IF NOT EXISTS)
    3. Load VectorStore (loads FAISS index from disk if it exists)

    Runs ONCE on shutdown:
    - Nothing needed; FAISS is saved after every write
    """
    settings.ensure_dirs()
    init_db(settings.db_path)

    vs = VectorStore(
        store_dir=settings.faiss_dir,
        model_name=settings.embedding_model,
    )
    set_vector_store(vs)
    print(f"✅ Agentic RAG started — {vs.total_vectors} vectors in index")

    yield   # ← app runs here

    print("🛑 Shutting down")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Agentic RAG API",
    version="1.0.0",
    docs_url="/api/docs",   # Swagger UI at /api/docs
    lifespan=lifespan,
)

# CORS — allow all origins in dev; restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(health.router)
app.include_router(documents.router)
app.include_router(sessions.router)
app.include_router(chat.router)

# ── Frontend ──────────────────────────────────────────────────────────────────

FRONTEND = Path("frontend")

if FRONTEND.exists():
    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(str(FRONTEND / "index.html"))
