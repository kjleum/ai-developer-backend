from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List

# Defaults make local import/dev easier; set real values in production.
DEFAULT_FERNET = "Xw1XZrZ9JZ4hZb3Xw1XZrZ9JZ4hZb3Xw1XZrZ9JZ4="

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/ai_platform"
    REDIS_URL: str = 'redis://localhost:6379/0'

    JWT_SECRET: str = "dev_jwt_secret_change_me"
    JWT_ALG: str = 'HS256'
    JWT_EXPIRES_MINUTES: int = 60 * 24 * 30  # 30 days

    ENCRYPTION_KEY: str = DEFAULT_FERNET

    CORS_ORIGINS: str = 'http://localhost,http://localhost:80'

    # Optional global provider keys
    OPENAI_API_KEY: str | None = None
    GROQ_API_KEY: str | None = None

    # Optional
    ADMIN_EMAIL: str | None = None
    ADMIN_PASSWORD: str | None = None
    TELEGRAM_BOT_TOKEN: str | None = None

    def cors_list(self) -> List[str]:
        return [o.strip() for o in (self.CORS_ORIGINS or '').split(',') if o.strip()]

settings = Settings()
