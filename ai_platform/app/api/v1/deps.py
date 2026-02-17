from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from typing import Optional

from app.core.security.auth import decode_jwt
from app.core.database import Database

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login", auto_error=False)
db = Database()

async def get_current_user(token: Optional[str] = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if not token:
        raise credentials_exception

    payload = decode_jwt(token)
    if payload is None:
        raise credentials_exception

    user = await db.get_user(payload["user_id"])
    if user is None:
        raise credentials_exception

    return user

async def get_current_admin(user: dict = Depends(get_current_user)):
    if not user.get("is_admin"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    return user

async def get_optional_user(token: Optional[str] = Depends(oauth2_scheme)):
    if not token:
        return None
    payload = decode_jwt(token)
    if not payload:
        return None
    return await db.get_user(payload["user_id"])
