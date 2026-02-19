from celery import Celery
import os
import asyncio
from app.core.database import Database

redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
celery_app = Celery("tasks", broker=redis_url, backend=redis_url)

db = Database()

@celery_app.task
def build_project_task(project_id: int, config: dict):
    loop = asyncio.get_event_loop()
    loop.run_until_complete(update_project_status(project_id, "building"))
    import time
    time.sleep(5)
    loop.run_until_complete(update_project_status(project_id, "live"))
    return {"status": "built"}

async def update_project_status(project_id: int, status: str):
    await db.update_project(project_id, {"status": status})
