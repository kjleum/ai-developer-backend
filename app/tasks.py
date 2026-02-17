from celery import Celery
import os

redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
celery_app = Celery("tasks", broker=redis_url, backend=redis_url)

@celery_app.task
def build_project_task(project_id: int, config: dict):
    """Background task to build a project."""
    # This would call the project builder, deploy engine, etc.
    print(f"Building project {project_id} with config {config}")
    return {"status": "built"}
