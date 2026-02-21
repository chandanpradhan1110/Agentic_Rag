"""
app/routers/documents.py

Endpoints:
  POST /api/documents/upload         ← HTML: handleFiles() → drop zone
  GET  /api/documents                ← HTML: renderFileList() + renderDataset()
  DELETE /api/documents/{id}         ← HTML: deleteDoc(id)
  POST /api/documents/rebuild-index  ← HTML: Settings panel rebuild button
"""
import uuid
from pathlib import Path

from fastapi import APIRouter, File, UploadFile, Depends, BackgroundTasks, HTTPException

from app.core.config import Settings, get_settings
from app.core.database import (
    create_document, list_documents, get_document, delete_document, update_doc_status
)
from app.services.background_tasks import process_document
from app.services.vector_store import VectorStore

router = APIRouter(prefix="/api/documents", tags=["documents"])

# Dependency: get shared VectorStore from app state
def _vs(settings: Settings = Depends(get_settings)):
    from app.dependencies import get_vector_store
    return get_vector_store()


@router.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
    vector_store: VectorStore = Depends(_vs),
):
    """
    WHAT THIS DOES:
    1. Validate extension (settings.allowed_extensions)
    2. Read file bytes, check size (settings.max_file_size_bytes)
    3. Save to UPLOAD_DIR/{doc_id}{ext}
    4. create_document() in DB with status="processing"
    5. background_tasks.add_task(process_document, ...)
    6. Return {doc_id, filename, status: "processing", message}

    WHY background task?
    Document loading + embedding can take 5-30 seconds for large PDFs.
    We return immediately (202) and process asynchronously.
    The frontend can poll GET /api/documents to see when status → "indexed".
    """
    ext = Path(file.filename).suffix.lower()
    if ext not in settings.allowed_extensions:
        raise HTTPException(
            415,
            f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(settings.allowed_extensions))}"
        )

    content = await file.read()
    if len(content) > settings.max_file_size_bytes:
        raise HTTPException(413, f"File exceeds {settings.max_file_size_mb} MB limit")

    doc_id = str(uuid.uuid4())
    ext_clean = ext.lstrip(".")
    save_path = settings.upload_dir / f"{doc_id}{ext}"
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    save_path.write_bytes(content)

    create_document(doc_id, file.filename, ext_clean.upper(), len(content))

    background_tasks.add_task(
        process_document,
        doc_id=doc_id,
        file_path=save_path,
        filename=file.filename,
        vector_store=vector_store,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    return {
        "doc_id":   doc_id,
        "filename": file.filename,
        "status":   "processing",
        "message":  "Document queued for indexing",
    }


@router.get("")
async def get_documents():
    """
    WHAT THIS DOES:
    Return all documents from DB.

    HTML uses this response to:
      - Render the file list in Upload panel (renderFileList)
      - Render the dataset table (renderDataset)
      - Update docCountBadge in Chat header

    Response shape matches what HTML expects:
      [{id, filename, file_type, file_size, chunk_count, status}]

    NOTE: HTML field mapping:
      doc.name   → doc.filename
      doc.type   → doc.file_type
      doc.chunks → doc.chunk_count
      doc.size   → doc.file_size
    """
    return list_documents()


@router.delete("/{doc_id}")
async def delete_doc(
    doc_id: str,
    settings: Settings = Depends(get_settings),
    vector_store: VectorStore = Depends(_vs),
):
    """
    WHAT THIS DOES:
    1. Check doc exists → 404 if not
    2. vector_store.delete_doc(doc_id) — soft-delete vectors
    3. Delete physical file from disk
    4. delete_document(doc_id) from DB
    5. Return {status: "deleted", doc_id}
    """
    doc = get_document(doc_id)
    if not doc:
        raise HTTPException(404, f"Document {doc_id} not found")

    vector_store.delete_doc(doc_id)

    for ext in settings.allowed_extensions:
        p = settings.upload_dir / f"{doc_id}{ext}"
        p.unlink(missing_ok=True)

    delete_document(doc_id)
    return {"status": "deleted", "doc_id": doc_id}


@router.post("/rebuild-index")
async def rebuild_index(vector_store: VectorStore = Depends(_vs)):
    """
    WHAT THIS DOES:
    Compact the FAISS index by rebuilding it without soft-deleted vectors.
    Call this after deleting many documents to reclaim memory.
    """
    size = vector_store.rebuild_index()
    return {"status": "rebuilt", "active_vectors": size}
