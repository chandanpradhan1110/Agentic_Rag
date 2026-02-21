"""
app/routers/sessions.py

Endpoints:
  POST /api/sessions                     ← HTML: newSession()
  GET  /api/sessions                     ← HTML: renderHistory()
  GET  /api/sessions/{id}/messages       ← HTML: loadSession(id)
  DELETE /api/sessions/{id}              ← (bonus: delete from history)
"""
import uuid
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.database import (
    create_session, list_sessions, get_session,
    delete_session, get_messages,
)

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


class SessionCreateBody(BaseModel):
    session_id: str | None = None
    title: str | None = None


@router.post("", status_code=201)
async def new_session(body: SessionCreateBody = SessionCreateBody()):
    """
    WHAT THIS DOES:
    HTML calls this on newSession() — when the user clicks "+ New Session".

    1. Use body.session_id or generate a new UUID
    2. create_session() in DB
    3. Return the session object

    HTML expects: {id, title, created_at, message_count}
    HTML sets:    sessionInfo.textContent = 'Session: ' + data.id.slice(0,8)
    """
    sid = body.session_id or str(uuid.uuid4())
    session = create_session(sid)
    return session


@router.get("")
async def get_sessions():
    """
    WHAT THIS DOES:
    HTML calls this to populate the History panel (renderHistory).

    Returns [{id, title, message_count, created_at}]

    HTML renders:
      history-item-title → s.title
      history-item-meta  → s.created_at + s.message_count
    """
    return list_sessions()


@router.get("/{session_id}/messages")
async def get_session_messages(session_id: str):
    """
    WHAT THIS DOES:
    HTML calls this in loadSession(id) — when user clicks a history item.

    Returns [{role, content, sources, created_at}]

    HTML re-renders the entire chat area from this response.
    """
    if not get_session(session_id):
        raise HTTPException(404, f"Session {session_id} not found")
    return get_messages(session_id)


@router.delete("/{session_id}", status_code=204)
async def remove_session(session_id: str):
    """
    WHAT THIS DOES:
    Delete a session and all its messages (CASCADE).
    """
    if not get_session(session_id):
        raise HTTPException(404, f"Session {session_id} not found")
    delete_session(session_id)
