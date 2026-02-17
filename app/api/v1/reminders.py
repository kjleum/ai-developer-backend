from fastapi import APIRouter, Depends, HTTPException
from datetime import datetime
from app.api.v1.deps import get_current_user
from app.core.database import Database

router = APIRouter()
db = Database()

@router.get("/reminders")
async def list_reminders(user: dict = Depends(get_current_user)):
    return {"reminders": await db.get_reminders(user["id"])}

@router.post("/reminders")
async def create_reminder(title: str, description: str = "", remind_at: str = "", user: dict = Depends(get_current_user)):
    if not title:
        raise HTTPException(400, "title required")
    try:
        dt = datetime.fromisoformat(remind_at) if remind_at else datetime.utcnow()
    except Exception:
        raise HTTPException(400, "bad remind_at (ISO)")
    rid = await db.create_reminder(user["id"], title, description, dt)
    return {"id": rid}

@router.delete("/reminders/{reminder_id}")
async def delete_reminder(reminder_id: int, user: dict = Depends(get_current_user)):
    await db.delete_reminder(user["id"], reminder_id)
    return {"ok": True}
