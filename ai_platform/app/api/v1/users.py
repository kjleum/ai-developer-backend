from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from pathlib import Path
import shutil

from app.api.v1.deps import get_current_user
from app.core.database import Database

router = APIRouter()
db = Database()

@router.get("/me")
async def get_me(user: dict = Depends(get_current_user)):
    return user

@router.put("/me/settings")
async def update_settings(settings: dict, user: dict = Depends(get_current_user)):
    await db.update_user_settings(user["id"], settings)
    return {"status": "ok"}

@router.post("/me/avatar")
async def upload_avatar(
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user)
):
    contents = await file.read()
    if len(contents) > 5 * 1024 * 1024:
        raise HTTPException(400, "File too large (max 5MB)")

    avatar_dir = Path("static/avatars")
    avatar_dir.mkdir(parents=True, exist_ok=True)

    file_extension = Path(file.filename).suffix
    avatar_path = avatar_dir / f"{user['id']}{file_extension}"

    with open(avatar_path, "wb") as f:
        f.write(contents)

    avatar_url = f"/static/avatars/{user['id']}{file_extension}"
    await db.update_user(user["id"], avatar_url=avatar_url)

    return {"avatar_url": avatar_url}
