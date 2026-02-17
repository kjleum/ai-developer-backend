import os
from typing import Optional, Dict, List
from datetime import datetime

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, update, delete, func

from app.models.db_models import Base, User, APIKey, Project, ProjectLog, Usage, Reminder, Notification, CalendarEvent
from app.core.security.auth import hash_password, verify_password
from app.core.security.encryption import encrypt_key, decrypt_key

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL not set")

ASYNC_DB_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")
engine = create_async_engine(ASYNC_DB_URL, echo=False, future=True)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

class Database:
    # ---------- Users ----------
    async def create_user(self, email: str, password: str, full_name: str = "") -> int:
        async with AsyncSessionLocal() as session:
            hashed = hash_password(password)
            user = User(email=email, password_hash=hashed, full_name=full_name)
            session.add(user)
            await session.commit()
            return user.id

    async def authenticate_user(self, email: str, password: str) -> Optional[Dict]:
        async with AsyncSessionLocal() as session:
            result = await session.execute(select(User).where(User.email == email))
            user = result.scalar_one_or_none()
            if user and verify_password(password, user.password_hash):
                user.last_login = func.now()
                await session.commit()
                return {"id": user.id, "email": user.email, "is_admin": user.is_admin}
            return None

    async def get_user(self, user_id: int) -> Optional[Dict]:
        async with AsyncSessionLocal() as session:
            user = await session.get(User, user_id)
            if user:
                return {
                    "id": user.id,
                    "email": user.email,
                    "full_name": user.full_name,
                    "avatar_url": user.avatar_url,
                    "phone": user.phone,
                    "telegram_id": user.telegram_id,
                    "is_admin": user.is_admin,
                    "balance": user.balance,
                    "total_projects": user.total_projects,
                    "total_tokens": user.total_tokens,
                    "settings": user.settings,
                    "created_at": user.created_at.isoformat() if user.created_at else None,
                    "last_login": user.last_login.isoformat() if user.last_login else None
                }
            return None

    async def update_user_settings(self, user_id: int, settings: dict):
        async with AsyncSessionLocal() as session:
            await session.execute(
                update(User).where(User.id == user_id).values(settings=settings)
            )
            await session.commit()

    async def update_user(self, user_id: int, **kwargs):
        async with AsyncSessionLocal() as session:
            await session.execute(
                update(User).where(User.id == user_id).values(**kwargs)
            )
            await session.commit()

    async def list_users(self) -> List[Dict]:
        async with AsyncSessionLocal() as session:
            result = await session.execute(select(User))
            users = result.scalars().all()
            return [
                {
                    "id": u.id,
                    "email": u.email,
                    "full_name": u.full_name,
                    "is_admin": u.is_admin,
                    "is_active": u.is_active,
                    "balance": u.balance,
                    "created_at": u.created_at.isoformat() if u.created_at else None
                }
                for u in users
            ]

    # ---------- API Keys ----------
    async def save_api_key(self, user_id: int, provider: str, api_key: str):
        async with AsyncSessionLocal() as session:
            encrypted = encrypt_key(api_key)
            key_entry = APIKey(user_id=user_id, provider=provider, encrypted_key=encrypted)
            session.add(key_entry)
            await session.commit()

    async def get_api_key(self, user_id: int, provider: str) -> Optional[str]:
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(APIKey).where(APIKey.user_id == user_id, APIKey.provider == provider)
            )
            key_entry = result.scalar_one_or_none()
            if key_entry:
                key_entry.last_used = func.now()
                await session.commit()
                return decrypt_key(key_entry.encrypted_key)
            return None

    async def list_api_keys(self, user_id: int) -> List[Dict]:
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(APIKey).where(APIKey.user_id == user_id)
            )
            keys = result.scalars().all()
            return [
                {
                    "provider": k.provider,
                    "created_at": k.created_at.isoformat() if k.created_at else None,
                    "last_used": k.last_used.isoformat() if k.last_used else None
                }
                for k in keys
            ]

    async def delete_api_key(self, user_id: int, provider: str):
        async with AsyncSessionLocal() as session:
            await session.execute(
                delete(APIKey).where(APIKey.user_id == user_id, APIKey.provider == provider)
            )
            await session.commit()

    # ---------- Projects ----------
    async def create_project(self, user_id: int, config: dict) -> int:
        async with AsyncSessionLocal() as session:
            proj = Project(
                user_id=user_id,
                name=config.get("name", "Untitled"),
                description=config.get("description", ""),
                config=config
            )
            session.add(proj)
            await session.commit()

            await session.execute(
                update(User).where(User.id == user_id).values(total_projects=User.total_projects + 1)
            )
            await session.commit()
            return proj.id

    async def get_project(self, project_id: int, user_id: int) -> Optional[Dict]:
        async with AsyncSessionLocal() as session:
            proj = await session.get(Project, project_id)
            if proj and proj.user_id == user_id:
                return {
                    "id": proj.id,
                    "name": proj.name,
                    "description": proj.description,
                    "status": proj.status,
                    "config": proj.config,
                    "files": proj.files,
                    "github_url": proj.github_url,
                    "deploy_url": proj.deploy_url,
                    "deploy_platform": proj.deploy_platform,
                    "created_at": proj.created_at.isoformat() if proj.created_at else None,
                    "updated_at": proj.updated_at.isoformat() if proj.updated_at else None
                }
            return None

    async def update_project(self, project_id: int, updates: dict):
        async with AsyncSessionLocal() as session:
            await session.execute(
                update(Project).where(Project.id == project_id).values(**updates)
            )
            await session.commit()

    async def delete_project(self, project_id: int, user_id: int):
        async with AsyncSessionLocal() as session:
            await session.execute(
                delete(Project).where(Project.id == project_id, Project.user_id == user_id)
            )
            await session.commit()

    async def list_projects(self, user_id: int) -> List[Dict]:
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(Project).where(Project.user_id == user_id).order_by(Project.created_at.desc())
            )
            projects = result.scalars().all()
            return [
                {
                    "id": p.id,
                    "name": p.name,
                    "status": p.status,
                    "created_at": p.created_at.isoformat() if p.created_at else None
                }
                for p in projects
            ]

    async def add_log(self, project_id: int, log_type: str, message: str):
        async with AsyncSessionLocal() as session:
            log = ProjectLog(project_id=project_id, log_type=log_type, message=message)
            session.add(log)
            await session.commit()

    # ---------- Usage ----------
    async def add_usage(self, user_id: int, provider: str, tokens: int, cost: float, endpoint: str = ""):
        async with AsyncSessionLocal() as session:
            usage = Usage(
                user_id=user_id,
                provider=provider,
                tokens_used=tokens,
                cost_estimate=cost,
                endpoint=endpoint
            )
            session.add(usage)
            await session.execute(
                update(User).where(User.id == user_id).values(total_tokens=User.total_tokens + tokens)
            )
            await session.commit()

    # ---------- Reminders ----------
    async def create_reminder(self, user_id: int, title: str, remind_at: datetime, description: str = "") -> int:
        async with AsyncSessionLocal() as session:
            rem = Reminder(
                user_id=user_id,
                title=title,
                description=description,
                remind_at=remind_at
            )
            session.add(rem)
            await session.commit()
            return rem.id

    async def get_reminders(self, user_id: int, active_only: bool = True) -> List[Dict]:
        async with AsyncSessionLocal() as session:
            query = select(Reminder).where(Reminder.user_id == user_id)
            if active_only:
                query = query.where(Reminder.is_active == True)
            result = await session.execute(query.order_by(Reminder.remind_at))
            reminders = result.scalars().all()
            return [
                {
                    "id": r.id,
                    "title": r.title,
                    "description": r.description,
                    "remind_at": r.remind_at.isoformat() if r.remind_at else None,
                    "is_active": r.is_active
                }
                for r in reminders
            ]

    async def delete_reminder(self, reminder_id: int, user_id: int):
        async with AsyncSessionLocal() as session:
            await session.execute(
                delete(Reminder).where(Reminder.id == reminder_id, Reminder.user_id == user_id)
            )
            await session.commit()

    # ---------- Notifications ----------
    async def create_notification(self, user_id: int, type: str, title: str, message: str):
        async with AsyncSessionLocal() as session:
            notif = Notification(user_id=user_id, type=type, title=title, message=message)
            session.add(notif)
            await session.commit()

    async def get_notifications(self, user_id: int, unread_only: bool = False) -> List[Dict]:
        async with AsyncSessionLocal() as session:
            query = select(Notification).where(Notification.user_id == user_id)
            if unread_only:
                query = query.where(Notification.is_read == False)
            result = await session.execute(query.order_by(Notification.created_at.desc()))
            notifs = result.scalars().all()
            return [
                {
                    "id": n.id,
                    "type": n.type,
                    "title": n.title,
                    "message": n.message,
                    "is_read": n.is_read,
                    "created_at": n.created_at.isoformat() if n.created_at else None
                }
                for n in notifs
            ]

    async def mark_notification_read(self, notif_id: int, user_id: int):
        async with AsyncSessionLocal() as session:
            await session.execute(
                update(Notification).where(Notification.id == notif_id, Notification.user_id == user_id).values(is_read=True)
            )
            await session.commit()

    # ---------- Calendar ----------
    async def create_event(self, user_id: int, event_data: dict) -> int:
        async with AsyncSessionLocal() as session:
            event = CalendarEvent(user_id=user_id, **event_data)
            session.add(event)
            await session.commit()
            return event.id

    async def get_events(self, user_id: int, start: datetime, end: datetime) -> List[Dict]:
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(CalendarEvent).where(
                    CalendarEvent.user_id == user_id,
                    CalendarEvent.start_time >= start,
                    CalendarEvent.start_time <= end
                )
            )
            events = result.scalars().all()
            return [
                {
                    "id": e.id,
                    "title": e.title,
                    "description": e.description,
                    "start_time": e.start_time.isoformat() if e.start_time else None,
                    "end_time": e.end_time.isoformat() if e.end_time else None,
                    "location": e.location
                }
                for e in events
            ]
