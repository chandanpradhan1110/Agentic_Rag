"""
app/services/vector_store.py

IMPLEMENT HERE:
- Thread-safe FAISS flat index with Sentence-Transformers embeddings
- Soft-delete (mark deleted, compact with rebuild_index)
- Persist index to disk so it survives restarts
"""
import pickle
import threading
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class VectorStore:
    """
    FAISS-backed vector store.

    Internal state:
      _index       : faiss.IndexFlatL2  — the actual vectors
      _meta        : list[dict|None]    — parallel list: meta for each vector
      _doc_map     : dict[doc_id, list[int]] — which positions belong to which doc
      _del_count   : int                — how many are soft-deleted
    """

    INDEX_FILE = "index.bin"
    META_FILE  = "meta.pkl"

    def __init__(self, store_dir: Path, model_name: str):
        self._lock = threading.RLock()
        self._dir  = store_dir
        self._dir.mkdir(parents=True, exist_ok=True)

        # STEP 1 — Load embedding model
        # IMPLEMENT: SentenceTransformer(model_name)
        self._model: SentenceTransformer = SentenceTransformer(model_name)
        self._dim: int = self._model.get_sentence_embedding_dimension()

        # STEP 2 — Initialize or load FAISS index
        # IMPLEMENT: faiss.IndexFlatL2(self._dim), then call self._load()
        self._index: faiss.IndexFlatL2 = faiss.IndexFlatL2(self._dim)
        self._meta: list[Optional[dict]] = []
        self._doc_map: dict[str, list[int]] = {}
        self._del_count: int = 0
        self._load()

    # ── Add chunks ────────────────────────────────────────────────────────────

    def add_chunks(self, doc_id: str, doc_name: str, chunks: list[str]) -> int:
        """
        IMPLEMENT:
        1. self._model.encode(chunks, normalize_embeddings=True) → numpy float32
        2. Under self._lock: record start = self._index.ntotal
        3. self._index.add(embeddings)
        4. Extend self._meta with dicts:
           {"doc_id": doc_id, "doc_name": doc_name, "chunk_index": i,
            "text": chunk, "deleted": False}
        5. Record positions in self._doc_map[doc_id]
        6. self._save()
        7. Return len(chunks)
        """
        if not chunks:
            return 0

        embs = self._model.encode(chunks, show_progress_bar=False, normalize_embeddings=True)
        embs = np.array(embs, dtype="float32")

        with self._lock:
            start = self._index.ntotal
            self._index.add(embs)
            positions = list(range(start, start + len(chunks)))
            self._doc_map[doc_id] = positions

            while len(self._meta) < start + len(chunks):
                self._meta.append(None)

            for i, (chunk, pos) in enumerate(zip(chunks, positions)):
                self._meta[pos] = {
                    "doc_id": doc_id,
                    "doc_name": doc_name,
                    "chunk_index": i,
                    "text": chunk,
                    "deleted": False,
                }
            self._save()

        return len(chunks)

    # ── Delete doc ────────────────────────────────────────────────────────────

    def delete_doc(self, doc_id: str):
        """
        IMPLEMENT soft-delete:
        1. For each pos in self._doc_map.pop(doc_id, []):
               self._meta[pos]["deleted"] = True
               self._del_count += 1
        2. self._save()
        """
        with self._lock:
            for pos in self._doc_map.pop(doc_id, []):
                if pos < len(self._meta) and self._meta[pos]:
                    self._meta[pos]["deleted"] = True
                    self._del_count += 1
            self._save()

    # ── Rebuild ───────────────────────────────────────────────────────────────

    def rebuild_index(self) -> int:
        """
        IMPLEMENT hard rebuild (removes deleted vectors permanently):
        1. Filter alive = [m for m in self._meta if m and not m["deleted"]]
        2. Re-embed all alive texts
        3. Create fresh faiss.IndexFlatL2(self._dim), add embeddings
        4. Rebuild self._meta and self._doc_map from alive list
        5. Reset self._del_count = 0
        6. self._save()
        7. Return self._index.ntotal
        """
        with self._lock:
            alive = [m for m in self._meta if m and not m.get("deleted")]
            if not alive:
                self._index = faiss.IndexFlatL2(self._dim)
                self._meta = []
                self._doc_map = {}
                self._del_count = 0
                self._save()
                return 0

            texts = [m["text"] for m in alive]
            embs = self._model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
            embs = np.array(embs, dtype="float32")

            self._index = faiss.IndexFlatL2(self._dim)
            self._index.add(embs)
            self._meta = alive
            self._doc_map = {}
            for pos, m in enumerate(alive):
                self._doc_map.setdefault(m["doc_id"], []).append(pos)
            self._del_count = 0
            self._save()

        return self._index.ntotal

    # ── Search ────────────────────────────────────────────────────────────────

    def search(self, query: str, k: int = 5) -> list[dict]:
        """
        IMPLEMENT:
        1. If index empty, return []
        2. Embed query with normalize_embeddings=True
        3. self._index.search(q_emb, min(k*4, ntotal))
        4. Walk (distance, idx) pairs:
              skip if idx < 0, or meta is None/deleted
              append {**meta, "score": float(dist)}
              stop when len(results) >= k
        5. Return results
        """
        with self._lock:
            if self._index.ntotal == 0:
                return []

            q = self._model.encode([query], show_progress_bar=False, normalize_embeddings=True)
            q = np.array(q, dtype="float32")
            fetch_k = min(k * 4, self._index.ntotal)
            dists, idxs = self._index.search(q, fetch_k)

            results = []
            for dist, idx in zip(dists[0], idxs[0]):
                if idx < 0 or idx >= len(self._meta):
                    continue
                m = self._meta[idx]
                if not m or m.get("deleted"):
                    continue
                results.append({**m, "score": float(dist)})
                if len(results) >= k:
                    break

        return results

    # ── Properties ───────────────────────────────────────────────────────────

    def has_documents(self) -> bool:
        return bool(self._doc_map)

    @property
    def total_vectors(self) -> int:
        return self._index.ntotal

    @property
    def active_vectors(self) -> int:
        return self._index.ntotal - self._del_count

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self):
        """
        IMPLEMENT:
        faiss.write_index(self._index, str(self._dir / self.INDEX_FILE))
        pickle.dump((self._meta, self._doc_map, self._del_count), open(META_FILE, "wb"))
        """
        faiss.write_index(self._index, str(self._dir / self.INDEX_FILE))
        with open(self._dir / self.META_FILE, "wb") as f:
            pickle.dump((self._meta, self._doc_map, self._del_count), f)

    def _load(self):
        """
        IMPLEMENT: If both files exist, load them.
        Wrap in try/except — if corrupt, start fresh.
        """
        idx_path  = self._dir / self.INDEX_FILE
        meta_path = self._dir / self.META_FILE
        if idx_path.exists() and meta_path.exists():
            try:
                self._index = faiss.read_index(str(idx_path))
                with open(meta_path, "rb") as f:
                    self._meta, self._doc_map, self._del_count = pickle.load(f)
            except Exception:
                pass   # start fresh
