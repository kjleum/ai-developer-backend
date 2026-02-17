from fastapi import APIRouter, Depends, HTTPException
from typing import List
from datetime import datetime

from app.api.v1.deps import get_current_user
from app.core.database import Database
from app.models.schemas import ReminderIn, ReminderOut

router = APIRouter()
db = Database()

@router.post("/reminders", response_model=dict)
async def create_reminder(
    reminder: ReminderIn,
    user: dict = Depends(get_current_user)
):
    rem_id = await db.create_reminder(
        user_id=user["id"],
        title=reminder.title,
        remind_at=reminder.remind_at,
        description=reminder.description
    )
    return {"id": rem_id, "status": "created"}

@router.get("/reminders", response_model=List[ReminderOut])
async def list_reminders(
    active_only: bool = True,
    user: dict = Depends(get_current_user)
):
    reminders = await db.get_reminders(user["id"], active_only)
    return reminders

@router.delete("/reminders/{reminder_id}")
async def delete_reminder(
    reminder_id: int,
    user: dict = Depends(get_current_user)
):
    await db.delete_reminder(reminder_id, user["id"])
    return {"status": "deleted"}
