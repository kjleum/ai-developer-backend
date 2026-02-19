from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import List

from app.api.v1.deps import get_current_user
from app.core.database import Database
from app.models.schemas import ProjectConfig, ProjectCreate

router = APIRouter()
db = Database()

@router.post("/projects/from-idea")
async def create_project_from_idea(
    idea: str,
    background_tasks: BackgroundTasks,
    user: dict = Depends(get_current_user)
):
    """Create a new project from a text idea."""
    # Simplified: In a real app, you'd analyze the idea with AI
    config = ProjectConfig(
        name=f"Project from idea",
        description=idea[:200],
        type="api"  # placeholder
    )

    project_id = await db.create_project(user["id"], config.dict())

    # Add background task to build the project
    # background_tasks.add_task(build_project_task.delay, project_id, config.dict())

    return {
        "project_id": project_id,
        "status": "created",
        "message": "Project created, build started"
    }

@router.get("/projects")
async def list_projects(user: dict = Depends(get_current_user)):
    projects = await db.list_projects(user["id"])
    return {"projects": projects}


@router.post("/projects")
async def create_project(payload: ProjectCreate, user: dict = Depends(get_current_user)):
    cfg = payload.config or {
        "name": payload.name,
        "description": payload.description,
        "type": "api",
        "features": [],
    }
    cfg.setdefault("name", payload.name)
    cfg.setdefault("description", payload.description)
    cfg.setdefault("type", "api")
    pid = await db.create_project(user["id"], cfg)
    return {"id": pid}

@router.get("/projects/{project_id}")
async def get_project(project_id: int, user: dict = Depends(get_current_user)):
    project = await db.get_project(project_id, user["id"])
    if not project:
        raise HTTPException(404, "Project not found")
    return project
