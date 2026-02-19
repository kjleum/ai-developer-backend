import base64
import hashlib
import hmac
import time
from typing import Any, Dict, Optional

import jwt
from passlib.context import CryptContext

from app.core.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def create_jwt(user_id: int, is_admin: bool = False) -> str:
    now = int(time.time())
    payload = {
        "user_id": user_id,
        "is_admin": bool(is_admin),
        "iat": now,
        "exp": now + settings.JWT_EXPIRES_MINUTES * 60,
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALG)


def decode_jwt(token: str) -> Optional[Dict[str, Any]]:
    try:
        return jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALG])
    except Exception:
        return None


# Telegram Mini App login
# Based on Telegram WebApp initData validation:
# https://core.telegram.org/bots/webapps#validating-data-received-via-the-mini-app

def verify_telegram_init_data(init_data: str, bot_token: str) -> Dict[str, str]:
    # Parse querystring into dict
    pairs = [p for p in init_data.split('&') if p]
    data: Dict[str, str] = {}
    for p in pairs:
        if '=' not in p:
            continue
        k, v = p.split('=', 1)
        data[k] = v

    received_hash = data.pop('hash', None)
    if not received_hash:
        raise ValueError('hash missing')

    # Create data_check_string
    items = [f"{k}={data[k]}" for k in sorted(data.keys())]
    data_check_string = "\n".join(items)

    # Secret key: HMAC_SHA256("WebAppData", bot_token)
    secret_key = hmac.new(b"WebAppData", bot_token.encode(), hashlib.sha256).digest()
    computed_hash = hmac.new(secret_key, data_check_string.encode(), hashlib.sha256).hexdigest()

    if not hmac.compare_digest(computed_hash, received_hash):
        raise ValueError('invalid init_data hash')

    return data
