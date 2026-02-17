from fastapi import APIRouter, Depends, HTTPException
from typing import List
from datetime import datetime
from pydantic import BaseModel

from app.api.v1.deps import get_current_user
from app.core.database import Database

class EventIn(BaseModel):
    title: str
    description: str = ""
    start_time: datetime
    end_time: datetime | None = None
    location: str = ""

router = APIRouter()
db = Database()

@router.post("/calendar/events")
async def create_event(
    event: EventIn,
    user: dict = Depends(get_current_user)
):
    event_id = await db.create_event(user["id"], event.dict())
    return {"id": event_id}

@router.get("/calendar/events")
async def list_events(
    start: datetime,
    end: datetime,
    user: dict = Depends(get_current_user)
):
    events = await db.get_events(user["id"], start, end)
    return {"events": events}
