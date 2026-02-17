import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.api.v1 import auth, users, projects, reminders, notifications, calendar, admin
from app.api.v1.ai import generate as ai_generate, keys as ai_keys
from app.core.database import Database
from app.models.db_models import Base
from app.core.database import engine

db = Database()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 AI Developer Platform запущен")
    # Ensure tables exist (works for small deployments; for prod use Alembic)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await db.ensure_admin_user()
    yield
    print("👋 Завершение работы")

app = FastAPI(
    title="AI Developer Platform",
    description="Полноценная AI-платформа (Web + Telegram Mini App)",
    version="6.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in origins if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("static/avatars", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(users.router, prefix="/api/v1/users", tags=["Users"])
app.include_router(projects.router, prefix="/api/v1/projects", tags=["Projects"])
app.include_router(ai_generate.router, prefix="/api/v1/ai", tags=["AI"])
app.include_router(ai_keys.router, prefix="/api/v1/ai", tags=["AI Keys"])

app.include_router(reminders.router, prefix="/api/v1", tags=["Reminders"])
app.include_router(notifications.router, prefix="/api/v1", tags=["Notifications"])
app.include_router(calendar.router, prefix="/api/v1", tags=["Calendar"])
app.include_router(admin.router, prefix="/api/v1/admin", tags=["Admin"])

@app.get("/health")
async def health_check():
    return {"status": "ok", "version": "6.0.0"}

@app.get("/")
async def root():
    return {"message": "AI Developer Platform API", "docs": "/api/docs", "version": "6.0.0"}
