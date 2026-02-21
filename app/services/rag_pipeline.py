"""
app/services/rag_pipeline.py

IMPLEMENT HERE:
LangGraph agentic RAG pipeline:
  retrieve → grade_chunks → [rewrite_query →]* generate → END

This is the core intelligence. Called by POST /api/chat.
"""
import asyncio
import time
from typing import TypedDict, Literal

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

from app.core.config import Settings


# ── State ─────────────────────────────────────────────────────────────────────

class RAGState(TypedDict):
    query: str
    rewritten_query: str
    retrieved_chunks: list[dict]
    relevant_chunks: list[dict]
    answer: str
    sources: list[str]
    rewrite_count: int
    needs_rewrite: bool


# ── LLM helper ────────────────────────────────────────────────────────────────

def _llm(settings: Settings, temperature: float = 0.0) -> ChatGroq:
    return ChatGroq(
        model=settings.groq_model,
        temperature=temperature,
        api_key=settings.groq_api_key,
        max_tokens=settings.groq_max_tokens,
    )


# ── Node: retrieve ────────────────────────────────────────────────────────────

def node_retrieve(state: RAGState, vector_store, top_k: int) -> RAGState:
    """
    IMPLEMENT:
    query = state["rewritten_query"] or state["query"]
    chunks = vector_store.search(query, k=top_k)
    return {**state, "retrieved_chunks": chunks}
    """
    query = state.get("rewritten_query") or state["query"]
    chunks = vector_store.search(query, k=top_k)
    return {**state, "retrieved_chunks": chunks}


# ── Node: grade_chunks ────────────────────────────────────────────────────────

def node_grade(state: RAGState, settings: Settings) -> RAGState:
    """
    IMPLEMENT:
    Ask the LLM which retrieved chunks are relevant to the query.

    Prompt the LLM with:
      - The user's query
      - Numbered list of chunk texts (truncated to 350 chars each)
    Ask for: comma-separated indices of relevant chunks, or "none"

    Parse the response:
      - If "none" → relevant_chunks=[], needs_rewrite=True (if retries remain)
      - Else → filter retrieved_chunks to those indices
      - On parse error → keep all chunks (safe fallback)

    WHEN to set needs_rewrite=True:
      len(relevant_chunks) == 0 AND state["rewrite_count"] < MAX_REWRITES
    """
    query = state.get("rewritten_query") or state["query"]
    chunks = state["retrieved_chunks"]
    max_rewrites = settings.max_rewrite_attempts

    if not chunks:
        return {**state, "relevant_chunks": [], "needs_rewrite": state["rewrite_count"] < max_rewrites}

    llm = _llm(settings)
    chunks_text = "\n---\n".join(f"[{i}] {c['text'][:350]}" for i, c in enumerate(chunks))

    prompt = (
        f"You are a relevance grader. Given this question and document chunks, "
        f"output ONLY a comma-separated list of relevant chunk indices (0-based), or 'none'.\n\n"
        f"Question: {query}\n\nChunks:\n{chunks_text}\n\nRelevant indices:"
    )

    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        raw = resp.content.strip().lower()
        if raw == "none":
            relevant = []
        else:
            idxs = [int(x.strip()) for x in raw.split(",") if x.strip().isdigit()]
            relevant = [chunks[i] for i in idxs if i < len(chunks)]
    except Exception:
        relevant = chunks   # fallback: keep all

    needs_rewrite = len(relevant) == 0 and state["rewrite_count"] < max_rewrites
    return {**state, "relevant_chunks": relevant, "needs_rewrite": needs_rewrite}


# ── Node: rewrite_query ───────────────────────────────────────────────────────

def node_rewrite(state: RAGState, settings: Settings) -> RAGState:
    """
    IMPLEMENT:
    Ask the LLM to rephrase the original query for better document retrieval.

    Prompt: "Rewrite this question to better match document terminology.
             Output ONLY the rewritten question."

    Return {**state, "rewritten_query": rewritten, "rewrite_count": count+1, "needs_rewrite": False}
    On error: return state with rewrite_count+1, skip rewrite
    """
    llm = _llm(settings, temperature=0.3)
    original = state["query"]

    prompt = (
        f"Rewrite this question to better match document terminology and improve retrieval. "
        f"Output ONLY the rewritten question.\n\nOriginal: {original}\nRewritten:"
    )

    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        rewritten = resp.content.strip()
    except Exception:
        rewritten = original   # fallback: keep original

    return {
        **state,
        "rewritten_query": rewritten,
        "rewrite_count": state["rewrite_count"] + 1,
        "needs_rewrite": False,
    }


# ── Node: generate ────────────────────────────────────────────────────────────

def node_generate(state: RAGState, settings: Settings) -> RAGState:
    """
    IMPLEMENT:
    Use relevant_chunks (fall back to retrieved_chunks if empty) to build context.

    Context format:
      [Source: doc_name, chunk #N]
      chunk text

    System prompt: "You are a document Q&A assistant. Answer using ONLY the context.
                   Cite document names. Say so if not found."

    Extract sources as unique set: "doc_name (chunk #N)"

    On error: return a graceful error message.
    """
    llm = _llm(settings, temperature=settings.groq_temperature)
    query = state.get("rewritten_query") or state["query"]
    chunks = state.get("relevant_chunks") or state.get("retrieved_chunks") or []

    if not chunks:
        return {
            **state,
            "answer": (
                "I couldn't find relevant information in your documents to answer this question. "
                "Try rephrasing, or upload documents that contain this information."
            ),
            "sources": [],
        }

    context = "\n\n---\n\n".join(
        f"[Source: {c['doc_name']}, chunk #{c['chunk_index']+1}]\n{c['text']}"
        for c in chunks
    )

    system = (
        "You are a helpful document Q&A assistant. "
        "Answer the question using ONLY the provided document context. "
        "Be concise and accurate. Cite the document name when referencing information. "
        "If the answer isn't in the context, say so clearly."
    )
    human = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

    try:
        resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
        answer = resp.content.strip()
    except Exception as e:
        raise RuntimeError(f"LLM generation failed: {e}")

    sources = list({f"{c['doc_name']} (chunk #{c['chunk_index']+1})" for c in chunks})
    return {**state, "answer": answer, "sources": sources}


# ── Routing ───────────────────────────────────────────────────────────────────

def _route(state: RAGState) -> Literal["rewrite_query", "generate"]:
    return "rewrite_query" if state.get("needs_rewrite") else "generate"


# ── Graph builder ─────────────────────────────────────────────────────────────

def _build_graph(vector_store, settings: Settings):
    """
    IMPLEMENT: Wire LangGraph nodes.

    Nodes:
      retrieve      → node_retrieve(state, vector_store, top_k)
      grade_chunks  → node_grade(state, settings)
      rewrite_query → node_rewrite(state, settings)
      generate      → node_generate(state, settings)

    Edges:
      retrieve → grade_chunks
      grade_chunks →[conditional]→ rewrite_query | generate
      rewrite_query → retrieve   (loop back!)
      generate → END
    """
    def retrieve(s): return node_retrieve(s, vector_store, settings.top_k_chunks)
    def grade(s):    return node_grade(s, settings)
    def rewrite(s):  return node_rewrite(s, settings)
    def generate(s): return node_generate(s, settings)

    g = StateGraph(RAGState)
    g.add_node("retrieve",      retrieve)
    g.add_node("grade_chunks",  grade)
    g.add_node("rewrite_query", rewrite)
    g.add_node("generate",      generate)

    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "grade_chunks")
    g.add_conditional_edges("grade_chunks", _route, {
        "rewrite_query": "rewrite_query",
        "generate":      "generate",
    })
    g.add_edge("rewrite_query", "retrieve")
    g.add_edge("generate", END)

    return g.compile()


# ── Public API ────────────────────────────────────────────────────────────────

async def run_rag(query: str, vector_store, settings: Settings) -> dict:
    """
    Entry point called by POST /api/chat.
    Runs the full pipeline in a thread pool (graph.invoke is synchronous).

    Returns:
      {answer, sources, rewritten_query, chunk_count, latency_ms}
    """
    start = time.perf_counter()
    graph = _build_graph(vector_store, settings)

    initial: RAGState = {
        "query": query,
        "rewritten_query": "",
        "retrieved_chunks": [],
        "relevant_chunks": [],
        "answer": "",
        "sources": [],
        "rewrite_count": 0,
        "needs_rewrite": False,
    }

    result = await asyncio.get_event_loop().run_in_executor(
        None, lambda: graph.invoke(initial)
    )

    latency_ms = int((time.perf_counter() - start) * 1000)
    return {
        "answer":           result["answer"],
        "sources":          result["sources"],
        "rewritten_query":  result.get("rewritten_query", ""),
        "chunk_count":      len(result.get("relevant_chunks") or result.get("retrieved_chunks", [])),
        "latency_ms":       latency_ms,
    }
