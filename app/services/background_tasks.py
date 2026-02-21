"""
app/services/background_tasks.py

IMPLEMENT HERE:
After a file is uploaded, FastAPI calls process_document() as a background task.
This runs in a thread pool (run_in_executor) so it doesn't block the event loop.
"""
import asyncio
from pathlib import Path

from app.core.database import update_doc_status
from app.services.document_loader import load_document, chunk_text


async def process_document(
    doc_id: str,
    file_path: Path,
    filename: str,
    vector_store,           # VectorStore instance
    chunk_size: int,
    chunk_overlap: int,
):
    """
    IMPLEMENT THIS FUNCTION — called after every successful file upload.

    Flow:
      1. await run_in_executor(load_document(file_path, filename))
         → raises ValueError if file is empty / unreadable
      2. await run_in_executor(chunk_text(text, chunk_size, chunk_overlap))
         → raises ValueError if no chunks produced
      3. await run_in_executor(vector_store.add_chunks(doc_id, filename, chunks))
         → returns chunk_count
      4. update_doc_status(doc_id, "indexed", chunk_count=chunk_count)

    On any exception:
      - update_doc_status(doc_id, "error")
      - log the error (print is fine for now)

    WHY run_in_executor?
      load_document and add_chunks are CPU/IO-bound (PDF parsing, FAISS operations).
      Running them in a thread pool keeps the async event loop free.
    """
    loop = asyncio.get_event_loop()
    try:
        # Step 1: Extract text
        text = await loop.run_in_executor(None, load_document, file_path, filename)
        if not text or len(text.strip()) < 10:
            raise ValueError("Document appears empty or has no extractable text")

        # Step 2: Chunk
        chunks = await loop.run_in_executor(None, chunk_text, text, chunk_size, chunk_overlap)
        if not chunks:
            raise ValueError("No chunks produced from document")

        # Step 3: Add to vector store (CPU-bound)
        chunk_count = await loop.run_in_executor(
            None,
            lambda: vector_store.add_chunks(doc_id, filename, chunks),
        )

        # Step 4: Mark indexed
        update_doc_status(doc_id, "indexed", chunk_count=chunk_count)
        print(f"[process_document] ✅ {filename} → {chunk_count} chunks indexed")

    except Exception as e:
        print(f"[process_document] ❌ {filename} failed: {e}")
        update_doc_status(doc_id, "error")
