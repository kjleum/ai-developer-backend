from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.core.database import init_db, Database
from app.api.v1.api import api_router

app = FastAPI(title="AI Developer Platform API", version="5.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_list() or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# API
app.include_router(api_router, prefix="/api/v1")

# Static (avatars etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/api/v1/health", tags=["health"])
async def health():
    return {"ok": True}


@app.on_event("startup")
async def on_startup():
    # Create tables for quick start (migrations recommended for production)
    await init_db()

    # Bootstrap admin if provided
    if settings.ADMIN_EMAIL and settings.ADMIN_PASSWORD:
        db = Database()
        # If admin exists, do nothing; otherwise create and flag as admin
        existing = await db.authenticate_user(settings.ADMIN_EMAIL, settings.ADMIN_PASSWORD)
        if not existing:
            try:
                admin_id = await db.create_user(settings.ADMIN_EMAIL, settings.ADMIN_PASSWORD, full_name="Admin")
                await db.update_user(admin_id, is_admin=True)
            except Exception:
                # likely already exists with different password
                pass
