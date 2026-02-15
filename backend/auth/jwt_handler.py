from jose import jwt
from datetime import datetime, timedelta
from backend.config import settings

ALGORITHM = "HS256"

def create_access_token(user_id: int):
    expire = datetime.utcnow() + timedelta(hours=24)
    payload = {"sub": str(user_id), "exp": expire}
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=ALGORITHM)

def decode_token(token: str):
    payload = jwt.decode(token, settings.JWT_SECRET, algorithms=[ALGORITHM])
    return int(payload.get("sub"))