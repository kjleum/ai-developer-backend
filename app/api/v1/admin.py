from fastapi import APIRouter, Depends, HTTPException
from typing import Optional

from app.api.v1.deps import get_current_admin
from app.core.database import Database

router = APIRouter()
db = Database()

@router.get("/users")
async def list_users(q: str = "", limit: int = 100, admin: dict = Depends(get_current_admin)):
    users = await db.list_users(q=q, limit=limit)
    return {"users": users}

@router.post("/users/{user_id}/flags")
async def set_user_flags(user_id: int, is_admin: Optional[bool] = None, is_active: Optional[bool] = None, admin: dict = Depends(get_current_admin)):
    if is_admin is None and is_active is None:
        raise HTTPException(400, "Nothing to update")
    await db.update_user(user_id, is_admin=is_admin if is_admin is not None else None, is_active=is_active if is_active is not None else None)
    return {"ok": True}

@router.get("/requests")
async def last_requests(limit: int = 200, admin: dict = Depends(get_current_admin)):
    # We expose latest usage rows as "requests log"
    items = await db.get_usage(limit=limit)
    return {"requests": items}

@router.get("/keys")
async def list_keys(user_id: Optional[int] = None, admin: dict = Depends(get_current_admin)):
    keys = await db.admin_list_api_keys(user_id=user_id)
    return {"keys": keys}
