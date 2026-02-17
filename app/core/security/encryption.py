from cryptography.fernet import Fernet
import os

ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY")
if not ENCRYPTION_KEY:
    raise ValueError("ENCRYPTION_KEY not set in environment variables")

cipher = Fernet(ENCRYPTION_KEY.encode())

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
