from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from app.core.database import Database
from app.core.security.auth import create_jwt, create_refresh_token
from app.models.schemas import UserRegister, Token

router = APIRouter()
db = Database()

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
