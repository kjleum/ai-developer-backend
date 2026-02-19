from cryptography.fernet import Fernet
from app.core.config import settings

cipher = Fernet(settings.ENCRYPTION_KEY.encode())

def encrypt_key(plain_text: str) -> str:
    if not plain_text:
        return ""
    return cipher.encrypt(plain_text.encode()).decode()

def decrypt_key(encrypted: str) -> str:
    if not encrypted:
        return ""
    return cipher.decrypt(encrypted.encode()).decode()

def mask_key(key: str) -> str:
    if not key or len(key) <= 8:
        return "******"
    return key[:4] + "..." + key[-4:]
