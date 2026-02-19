from fastapi import APIRouter, Depends, HTTPException

from app.api.v1.deps import get_current_user
from app.core.database import Database
from app.core.ai.manager import AIManager
from app.models.schemas import ChatThreadCreate, ChatMessageCreate


router = APIRouter()
db = Database()
ai = AIManager()


@router.get("/chat/threads")
async def list_threads(project_id: int | None = None, limit: int = 100, user: dict = Depends(get_current_user)):
    return {"threads": await db.list_threads(user["id"], project_id=project_id, limit=limit)}


@router.post("/chat/threads")
async def create_thread(payload: ChatThreadCreate, user: dict = Depends(get_current_user)):
    try:
        tid = await db.create_thread(user["id"], payload.project_id, payload.title)
    except ValueError:
        raise HTTPException(404, "Project not found")
    return {"id": tid}


@router.get("/chat/threads/{thread_id}")
async def get_thread(thread_id: int, user: dict = Depends(get_current_user)):
    th = await db.get_thread(user["id"], thread_id)
    if not th:
        raise HTTPException(404, "Thread not found")
    return th


@router.get("/chat/threads/{thread_id}/messages")
async def list_messages(thread_id: int, limit: int = 200, user: dict = Depends(get_current_user)):
    try:
        msgs = await db.list_messages(user["id"], thread_id, limit=limit)
    except ValueError:
        raise HTTPException(404, "Thread not found")
    return {"messages": msgs}


@router.post("/chat/threads/{thread_id}/messages")
async def post_message(thread_id: int, payload: ChatMessageCreate, user: dict = Depends(get_current_user)):
    """Append a message. If role=user, we will also generate assistant reply and save it."""
    if not payload.content.strip():
        raise HTTPException(400, "content required")

    try:
        await db.add_message(user["id"], thread_id, payload.role, payload.content)
    except ValueError:
        raise HTTPException(404, "Thread not found")

    if payload.role != "user":
        return {"ok": True}

    # Generate assistant response
    resp = await ai.generate(
        prompt=payload.content,
        provider=payload.provider,
        model=payload.model,
        temperature=payload.temperature,
        max_tokens=payload.max_tokens,
        json_mode=False,
        user_id=user["id"],
    )
    await db.add_message(user["id"], thread_id, "assistant", resp)
    return {"response": resp}
