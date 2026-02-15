from cryptography.fernet import Fernet
from backend.config import settings

fernet = Fernet(settings.MASTER_ENCRYPTION_KEY.encode())

def encrypt_key(key: str):
    return fernet.encrypt(key.encode()).decode()

def decrypt_key(encrypted: str):
    return fernet.decrypt(encrypted.encode()).decode()