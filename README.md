# Agentic RAG — FastAPI Backend Implementation Guide

## What the HTML Currently Does (The Problem)

The uploaded `agentic-rag.html` is a **pure frontend demo** with ZERO real backend calls:

| Feature         | What HTML does TODAY                               | What it SHOULD do                           |
| --------------- | -------------------------------------------------- | ------------------------------------------- |
| File Upload     | `FileReader` in browser, fake chunk count        | POST to `/api/documents/upload`           |
| Documents list  | In-memory `documents[]` JS array                 | GET `/api/documents`                      |
| Delete doc      | Splice from JS array                               | DELETE `/api/documents/{id}`              |
| Chat            | Calls Anthropic API**directly from browser** | POST `/api/chat` (backend proxies to LLM) |
| Sessions        | In-memory `sessions[]` JS array                  | POST/GET `/api/sessions`                  |
| History         | Rendered from same JS array                        | GET `/api/sessions/{id}/messages`         |
| Stats (Dataset) | Computed from JS array                             | GET `/api/stats`                          |
| Settings        | API key saved to `let apiKey` variable           | Stays frontend (key goes in .env)           |

---

## Project Structure

```
AGENTIC_RAG/
│
├── main.py                         ← FastAPI app entry point
├── .env                            ← GROQ_API_KEY, settings (never commit)
├── .env.example                    ← Template for .env
├── requirements.txt
├── pyproject.toml
│
├── app/
│   ├── __init__.py
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py               ← All settings (pydantic-settings)
│   │   └── database.py             ← SQLite init + all DB operations
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── vector_store.py         ← FAISS index: add/search/delete chunks
│   │   ├── document_loader.py      ← Load PDF/TXT/DOCX/CSV → text → chunks
│   │   ├── rag_pipeline.py         ← LangGraph: retrieve→grade→rewrite→generate
│   │   └── background_tasks.py     ← Async doc indexing after upload
│   │
│   └── routers/
│       ├── __init__.py
│       ├── documents.py            ← POST /upload, GET /, DELETE /{id}
│       ├── sessions.py             ← POST /, GET /, GET /{id}/messages
│       ├── chat.py                 ← POST /chat
│       └── health.py               ← GET /health, GET /stats
│
├── frontend/
│   └── index.html                  ← agentic-rag.html (updated to call backend)
│
├── data/                           ← Auto-created at runtime (gitignored)
│   ├── rag.db                      ← SQLite database
│   ├── uploads/                    ← Raw uploaded files
│   ├── faiss/                      ← FAISS index + metadata pickle
│   └── logs/                       ← Rotating log files
│
└── tests/
    └── test_api.py
```

---

## Implementation Order — Do This Sequence

### STEP 1 — `app/core/config.py`

**Implement first** because every other file imports settings.

What to put here:

- `GROQ_API_KEY` from env
- `GROQ_MODEL` (default: llama-3.1-8b-instant)
- `EMBEDDING_MODEL` (default: all-MiniLM-L6-v2)
- `CHUNK_SIZE`, `CHUNK_OVERLAP`
- `TOP_K_CHUNKS`
- File paths: `DB_PATH`, `UPLOAD_DIR`, `FAISS_DIR`, `LOG_DIR`
- `MAX_FILE_SIZE_MB`, `ALLOWED_EXTENSIONS`

---

### STEP 2 — `app/core/database.py`

**Implement second** because routers need DB functions.

Tables to create:

```sql
sessions(id TEXT PK, title TEXT, created_at TEXT, message_count INT)
messages(id INT PK, session_id TEXT FK, role TEXT, content TEXT, sources TEXT, created_at TEXT)
documents(id TEXT PK, filename TEXT, file_type TEXT, file_size INT, chunk_count INT, status TEXT, uploaded_at TEXT)
```

Functions to implement:

- `init_db()` — CREATE TABLE IF NOT EXISTS
- `create_session(id)` / `get_session(id)` / `list_sessions()` / `delete_session(id)`
- `add_message(session_id, role, content, sources)` / `get_messages(session_id)`
- `create_document(...)` / `get_document(id)` / `list_documents()` / `delete_document(id)` / `update_doc_status(...)`
- `get_stats()` → `{total_documents, total_chunks, total_size_bytes}`

---

### STEP 3 — `app/services/document_loader.py`

**Implement third** — needed by background_tasks.py.

Functions:

- `load_document(path, filename) → str` — dispatch by extension
  - `.pdf` → pypdf PdfReader
  - `.txt/.md` → read_text()
  - `.docx` → python-docx paragraphs
  - `.csv` → pandas to string rows
- `chunk_text(text, size, overlap) → list[str]` — sliding word window

---

### STEP 4 — `app/services/vector_store.py`

**Implement fourth** — needed by rag_pipeline and documents router.

Class `VectorStore`:

- `__init__()` — load SentenceTransformer, load or create FAISS flat index, load metadata pickle
- `add_chunks(doc_id, doc_name, chunks) → int` — embed + add to index
- `delete_doc(doc_id)` — soft delete (mark metadata deleted=True)
- `search(query, k=5) → list[dict]` — embed query, search, filter deleted
- `rebuild_index()` — hard rebuild removing soft-deleted vectors
- `_save() / _load()` — faiss.write_index / pickle metadata

---

### STEP 5 — `app/services/background_tasks.py`

**Implement fifth** — called by documents router after upload.

One async function:

```python
async def process_document(doc_id, file_path, filename, vector_store, chunk_size, chunk_overlap)
    # 1. load_document() → text
    # 2. chunk_text() → chunks
    # 3. vector_store.add_chunks()
    # 4. update_doc_status(doc_id, "indexed", chunk_count=N)
```

---

### STEP 6 — `app/services/rag_pipeline.py`

**Implement sixth** — called by chat router.

LangGraph pipeline:

```
retrieve → grade_chunks → [rewrite_query →]* generate → END
```

- Node `retrieve`: `vector_store.search(query)`
- Node `grade_chunks`: LLM scores each chunk 0–1, filters below threshold
- Node `rewrite_query`: LLM rewrites query if no relevant chunks found (max 2 retries)
- Node `generate`: LLM answers from relevant chunks with system prompt

Public function:

```python
async def run_rag(query, session_id, vector_store, settings) -> dict:
    # returns {answer, sources, rewritten_query, chunk_count}
```

---

### STEP 7 — `app/routers/documents.py`

**Implement seventh**.

Endpoints needed by the HTML:

```
POST /api/documents/upload
  - Accept UploadFile
  - Validate extension + size
  - Save to UPLOAD_DIR/{doc_id}{ext}
  - create_document() in DB with status="processing"
  - Add background_task: process_document()
  - Return {doc_id, filename, status}

GET /api/documents
  - list_documents() from DB
  - Return list of {id, filename, file_type, file_size, chunk_count, status}
  ↑ HTML needs: id, name(=filename), size, type, chunks(=chunk_count)

DELETE /api/documents/{doc_id}
  - vector_store.delete_doc(doc_id)
  - Delete file from disk
  - delete_document() from DB
  - Return {status: "deleted"}

POST /api/documents/rebuild-index
  - vector_store.rebuild_index()
  - Return {active_vectors: N}
```

---

### STEP 8 — `app/routers/sessions.py`

```
POST /api/sessions
  Body: {session_id?: str}
  - create_session() in DB
  - Return session object

GET /api/sessions
  - list_sessions() from DB
  - Return [{id, title, message_count, created_at}]
  ↑ HTML renders: title, time(created_at), messages.length(message_count)

GET /api/sessions/{session_id}/messages
  - get_messages(session_id)
  - Return [{role, content, sources, created_at}]

DELETE /api/sessions/{session_id}
  - delete_session(session_id) (CASCADE deletes messages)
```

---

### STEP 9 — `app/routers/chat.py`

```
POST /api/chat
  Body: {session_id: str, message: str}
  1. If no documents indexed → 400 error
  2. ensure_session(session_id)
  3. add_message(session_id, "user", message)
  4. result = await run_rag(message, vector_store, settings)
  5. add_message(session_id, "assistant", result.answer, result.sources)
  6. Return {answer, sources, rewritten_query, chunk_count}
```

---

### STEP 10 — `app/routers/health.py`

```
GET /api/health
  Return {status, groq_model, documents_indexed, vector_index_size}

GET /api/stats
  Return {total_documents, total_chunks, total_size_bytes}
  ↑ HTML Dataset panel shows: statDocs, statChunks, statSize
```

---

### STEP 11 — `main.py`

Wire everything together:

- lifespan: init_db(), create VectorStore
- Add CORS middleware
- Mount routers: /api/documents, /api/sessions, /api/chat, /api/health
- StaticFiles for frontend/
- SPA fallback: GET / → serve index.html

---

### STEP 12 — Update `frontend/index.html`

Replace all in-memory operations with fetch() calls:

| HTML function                           | Replace with                                             |
| --------------------------------------- | -------------------------------------------------------- |
| `handleFiles()`                       | `fetch('POST /api/documents/upload', FormData)`        |
| `deleteDoc(id)`                       | `fetch('DELETE /api/documents/{id}')`                  |
| `renderFileList()`                    | `fetch('GET /api/documents')` then render              |
| `renderDataset()`                     | `fetch('GET /api/stats')` then render                  |
| `sendMessage()` / `callClaudeAPI()` | `fetch('POST /api/chat', {session_id, message})`       |
| `newSession()`                        | `fetch('POST /api/sessions')`                          |
| `renderHistory()`                     | `fetch('GET /api/sessions')` then render               |
| `loadSession(id)`                     | `fetch('GET /api/sessions/{id}/messages')` then render |

---

## API Contract (exactly what HTML expects)

### GET /api/documents

```json
[
  {
    "id": "uuid",
    "filename": "report.pdf",
    "file_type": "PDF",
    "file_size": 204800,
    "chunk_count": 42,
    "status": "indexed"
  }
]
```

### POST /api/chat

Request:

```json
{ "session_id": "abc123", "message": "What does the report say about Q3?" }
```

Response:

```json
{
  "answer": "According to the report...",
  "sources": ["report.pdf (chunk #3)", "report.pdf (chunk #7)"],
  "rewritten_query": "",
  "chunk_count": 5
}
```

### GET /api/sessions

```json
[
  { "id": "abc123", "title": "First question...", "message_count": 4, "created_at": "2026-02-21T..." }
]
```

### GET /api/stats

```json
{ "total_documents": 3, "total_chunks": 127, "total_size_bytes": 614400 }
```
