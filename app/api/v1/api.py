from fastapi import APIRouter

from app.api.v1 import auth, users, keys, projects, generate, reminders, notifications, calendar, admin, chat

api_router = APIRouter()

api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(keys.router, tags=["keys"])
api_router.include_router(projects.router, tags=["projects"])
api_router.include_router(generate.router, prefix="/ai", tags=["ai"])
api_router.include_router(reminders.router, prefix="/reminders", tags=["reminders"])
api_router.include_router(notifications.router, prefix="/notifications", tags=["notifications"])
api_router.include_router(calendar.router, prefix="/calendar", tags=["calendar"])
api_router.include_router(admin.router, prefix="/admin", tags=["admin"])
api_router.include_router(chat.router, tags=["chat"])
