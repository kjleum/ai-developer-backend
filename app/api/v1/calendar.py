from fastapi import APIRouter, Depends, HTTPException
from datetime import datetime
from app.api.v1.deps import get_current_user
from app.core.database import Database

router = APIRouter()
db = Database()

@router.get("/calendar/events")
async def list_events(user: dict = Depends(get_current_user)):
    return {"events": await db.get_events(user["id"])}

@router.post("/calendar/events")
async def create_event(title: str, start_time: str, end_time: str = "", description: str = "", location: str = "", user: dict = Depends(get_current_user)):
    if not title:
        raise HTTPException(400, "title required")
    try:
        st = datetime.fromisoformat(start_time)
        et = datetime.fromisoformat(end_time) if end_time else None
    except Exception:
        raise HTTPException(400, "bad datetime ISO")
    eid = await db.create_event(user["id"], title, description, st, et, location)
    return {"id": eid}
