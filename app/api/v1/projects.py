from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel

from app.api.v1.deps import get_current_user
from app.core.database import Database
from app.models.schemas import ProjectConfig
from app.tasks import build_project_task

router = APIRouter()
db = Database()

class ProjectOut(BaseModel):
    id: int
    name: str
    description: Optional[str]
    status: str
    config: dict
    files: dict
    github_url: Optional[str]
    deploy_url: Optional[str]
    created_at: datetime
    updated_at: Optional[datetime]

@router.post("/projects", response_model=dict)
async def create_project(
    config: ProjectConfig,
    background_tasks: BackgroundTasks,
    user: dict = Depends(get_current_user)
):
    project_id = await db.create_project(user["id"], config.dict())
    background_tasks.add_task(build_project_task.delay, project_id, config.dict())
    return {"project_id": project_id, "status": "created"}

@router.get("/projects", response_model=List[ProjectOut])
async def list_projects(user: dict = Depends(get_current_user)):
    projects = await db.list_projects(user["id"])
    return projects

@router.get("/projects/{project_id}", response_model=ProjectOut)
async def get_project(project_id: int, user: dict = Depends(get_current_user)):
    project = await db.get_project(project_id, user["id"])
    if not project:
        raise HTTPException(404, "Project not found")
    return project

@router.put("/projects/{project_id}")
async def update_project(
    project_id: int,
    updates: dict,
    user: dict = Depends(get_current_user)
):
    project = await db.get_project(project_id, user["id"])
    if not project:
        raise HTTPException(404, "Project not found")
    await db.update_project(project_id, updates)
    return {"status": "updated"}

@router.delete("/projects/{project_id}")
async def delete_project(project_id: int, user: dict = Depends(get_current_user)):
    await db.delete_project(project_id, user["id"])
    return {"status": "deleted"}

@router.post("/projects/from-idea")
async def create_project_from_idea(
    idea: str,
    background_tasks: BackgroundTasks,
    user: dict = Depends(get_current_user)
):
    config = ProjectConfig(
        name=f"Project from idea",
        description=idea[:200],
        type="api"
    )
    project_id = await db.create_project(user["id"], config.dict())
    background_tasks.add_task(build_project_task.delay, project_id, config.dict())
    return {"project_id": project_id, "status": "created"}
