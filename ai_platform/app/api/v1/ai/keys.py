from fastapi import APIRouter, Depends, HTTPException

from app.api.v1.deps import get_current_user
from app.core.database import Database
from app.core.security.encryption import mask_key
from app.models.schemas import APIKeyIn, APIKeyOut

router = APIRouter()
db = Database()

@router.post("/keys", response_model=dict)
async def add_api_key(
    key_data: APIKeyIn,
    user: dict = Depends(get_current_user)
):
    await db.save_api_key(user["id"], key_data.provider, key_data.api_key)
    return {"status": "ok", "provider": key_data.provider}

@router.get("/keys", response_model=list[APIKeyOut])
async def list_api_keys(user: dict = Depends(get_current_user)):
    keys = await db.list_api_keys(user["id"])
    return keys

@router.delete("/keys/{provider}")
async def delete_api_key(
    provider: str,
    user: dict = Depends(get_current_user)
):
    await db.delete_api_key(user["id"], provider)
    return {"status": "ok"}
