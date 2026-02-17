from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
import os, json

from app.core.database import Database
from app.core.security.auth import create_jwt, verify_telegram_init_data
from app.models.schemas import UserRegister, Token

router = APIRouter()
db = Database()

class LoginJSON(BaseModel):
    email: EmailStr
    password: str

class TelegramLogin(BaseModel):
    init_data: str  # window.Telegram.WebApp.initData

@router.post("/register", response_model=Token)
async def register(user_data: UserRegister):
    user_id = await db.create_user(
        email=user_data.email,
        password=user_data.password,
        full_name=user_data.full_name
    )
    access_token = create_jwt(user_id)
    return {"access_token": access_token, "token_type": "bearer"}

@router.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await db.authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_jwt(user["id"], user["is_admin"])
    return {"access_token": access_token, "token_type": "bearer"}

@router.post("/login_json", response_model=Token)
async def login_json(payload: LoginJSON):
    user = await db.authenticate_user(payload.email, payload.password)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    access_token = create_jwt(user["id"], user["is_admin"])
    return {"access_token": access_token, "token_type": "bearer"}

@router.post("/telegram", response_model=Token)
async def telegram(payload: TelegramLogin):
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not bot_token:
        raise HTTPException(500, "TELEGRAM_BOT_TOKEN not set")

    parsed = verify_telegram_init_data(payload.init_data, bot_token)
    tg_user = json.loads(parsed.get("user", "{}") or "{}")
    tg_id = str(tg_user.get("id") or "")
    if not tg_id:
        raise HTTPException(400, "telegram user not found")

    user = await db.get_user_by_telegram_id(tg_id)
    if not user:
        email = f"tg_{tg_id}@telegram.local"
        full_name = (tg_user.get("first_name","") + " " + tg_user.get("last_name","")).strip()
        user_id = await db.create_user(email=email, password=os.urandom(16).hex(), full_name=full_name)
        await db.update_user(user_id, telegram_id=tg_id)
        access_token = create_jwt(user_id, False)
        return {"access_token": access_token, "token_type": "bearer"}

    access_token = create_jwt(user["id"], user["is_admin"])
    return {"access_token": access_token, "token_type": "bearer"}
