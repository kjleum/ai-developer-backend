from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DATABASE_URL: str
    JWT_SECRET: str
    MASTER_ENCRYPTION_KEY: str
    ENV: str = "development"

    class Config:
        env_file = ".env"

settings = Settings()