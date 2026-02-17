from fastapi import APIRouter, Depends
from typing import List

from app.api.v1.deps import get_current_user
from app.core.database import Database

router = APIRouter()
db = Database()

@router.get("/notifications")
async def get_notifications(
    unread_only: bool = False,
    user: dict = Depends(get_current_user)
):
    notifs = await db.get_notifications(user["id"], unread_only)
    return {"notifications": notifs}

@router.post("/notifications/{notif_id}/read")
async def mark_notification_read(
    notif_id: int,
    user: dict = Depends(get_current_user)
):
    await db.mark_notification_read(notif_id, user["id"])
    return {"status": "marked as read"}
