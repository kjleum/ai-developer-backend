from fastapi import APIRouter, Depends
from app.api.v1.deps import get_current_user
from app.core.database import Database

router = APIRouter()
db = Database()

@router.get("/notifications")
async def list_notifications(user: dict = Depends(get_current_user)):
    return {"notifications": await db.get_notifications(user["id"])}

@router.post("/notifications/{notification_id}/read")
async def mark_read(notification_id: int, user: dict = Depends(get_current_user)):
    await db.mark_notification_read(user["id"], notification_id)
    return {"ok": True}
