
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import os
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

# Импортируем наши модули
from core.ai_core import AIAgent
from core.code_generator import CodeGenerator
from core.deploy_engine import DeployEngine
from core.project_manager import ProjectManager

app = FastAPI(
    title="AI Developer API",
    description="AI-разработчик для создания любых проектов",
    version="1.0.0"
)

# CORS для работы с Telegram Mini App
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене указать конкретный домен
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Инициализация компонентов
ai_agent = AIAgent()
code_gen = CodeGenerator()
deployer = DeployEngine()
project_db = ProjectManager()

# Модели данных
class CreateProjectRequest(BaseModel):
    description: str
    user_id: str
    project_name: Optional[str] = None

class UpdateProjectRequest(BaseModel):
    project_id: str
    feedback: str
    user_id: str

class ProjectStatus(BaseModel):
    project_id: str
    user_id: str

@app.get("/")
async def root():
    return {"status": "AI Developer API работает", "version": "1.0.0"}

@app.post("/create")
async def create_project(request: CreateProjectRequest, background_tasks: BackgroundTasks):
    """
    Создание нового проекта:
    1. AI анализирует описание
    2. Генерирует архитектуру и код
    3. Сохраняет в базу
    4. Деплоит (в фоне)
    """
    try:
        # Шаг 1: AI анализирует задачу
        print(f"[1/5] Анализ проекта: {request.description[:50]}...")
        architecture = await ai_agent.analyze_project(request.description)

        # Шаг 2: Генерация кода
        print(f"[2/5] Генерация кода для {architecture['type']}...")
        files = await code_gen.generate(architecture)

        # Шаг 3: Сохранение в базу
        print("[3/5] Сохранение проекта...")
        project_id = await project_db.create(
            user_id=request.user_id,
            name=request.project_name or f"project_{architecture['type']}",
            description=request.description,
            architecture=architecture,
            files=files,
            status="generating"
        )

        # Шаг 4: Запускаем деплой в фоне
        print("[4/5] Запуск деплоя...")
        background_tasks.add_task(deploy_project_task, project_id, files)

        return {
            "success": True,
            "project_id": project_id,
            "status": "generating",
            "architecture": architecture,
            "message": "Проект создан, начат деплой. Проверь статус через /status"
        }

    except Exception as e:
        print(f"Ошибка: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def deploy_project_task(project_id: str, files: Dict[str, str]):
    """Фоновая задача деплоя"""
    try:
        # Деплой
        deploy_result = await deployer.deploy(project_id, files)

        # Обновляем статус
        await project_db.update_status(
            project_id=project_id,
            status="live" if deploy_result["success"] else "error",
            url=deploy_result.get("url"),
            logs=deploy_result.get("logs")
        )

    except Exception as e:
        await project_db.update_status(
            project_id=project_id,
            status="error",
            logs=str(e)
        )

@app.get("/status/{project_id}")
async def get_status(project_id: str, user_id: str):
    """Проверка статуса проекта"""
    project = await project_db.get(project_id, user_id)
    if not project:
        raise HTTPException(status_code=404, detail="Проект не найден")

    return {
        "project_id": project_id,
        "status": project["status"],
        "url": project.get("url"),
        "name": project["name"],
        "created_at": project["created_at"],
        "logs": project.get("logs", "")
    }

@app.post("/update")
async def update_project(request: UpdateProjectRequest):
    """Обновление проекта по фидбеку"""
    try:
        # Получаем текущий проект
        project = await project_db.get(request.project_id, request.user_id)
        if not project:
            raise HTTPException(status_code=404, detail="Проект не найден")

        # AI анализирует фидбек и текущий код
        print(f"Исправление проекта по фидбеку: {request.feedback}")
        new_files = await ai_agent.fix_code(
            current_files=project["files"],
            architecture=project["architecture"],
            feedback=request.feedback
        )

        # Передеплой
        deploy_result = await deployer.redeploy(request.project_id, new_files)

        # Обновляем в базе
        await project_db.update_files(request.project_id, new_files)
        await project_db.update_status(
            project_id=request.project_id,
            status="live" if deploy_result["success"] else "error",
            url=deploy_result.get("url"),
            logs=deploy_result.get("logs")
        )

        return {
            "success": True,
            "url": deploy_result.get("url"),
            "status": "updated"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/projects/{user_id}")
async def list_projects(user_id: str):
    """Список всех проектов пользователя"""
    projects = await project_db.list(user_id)
    return {"projects": projects}

@app.delete("/project/{project_id}")
async def delete_project(project_id: str, user_id: str):
    """Удаление проекта"""
    await project_db.delete(project_id, user_id)
    # TODO: удалить с Render и GitHub
    return {"success": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
