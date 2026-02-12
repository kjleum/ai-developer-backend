from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv

load_dotenv()

from models.schemas import *
from core.ai_manager import AIManager
from core.project_builder import ProjectBuilder
from core.deploy_engine import DeployEngine
from core.database import Database

# Глобальные объекты
ai_manager = AIManager()
project_builder = ProjectBuilder()
deploy_engine = DeployEngine()
db = Database()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Инициализация при старте"""
    print("🚀 AI Developer Platform запущен")
    yield
    print("👋 Завершение работы")

app = FastAPI(
    title="AI Developer Platform",
    description="Продвинутая платформа для создания проектов с AI",
    version="2.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ AI ПРОВАЙДЕРЫ ============

@app.get("/ai/providers")
async def get_ai_providers():
    """Получить список доступных AI провайдеров"""
    return {
        "providers": ai_manager.get_available_providers(),
        "recommended": "groq"
    }

@app.get("/ai/providers/{provider}/models")
async def get_ai_models(provider: str):
    """Получить модели провайдера"""
    return {"models": ai_manager.get_models(provider)}

# ============ ПРИМЕРЫ ПРОЕКТОВ ============

@app.get("/examples")
async def get_examples():
    """Получить динамические примеры проектов"""
    import random

    all_examples = [
        {
            "id": "tiktok_scraper",
            "title": "📱 TikTok Парсер",
            "description": "Собирает видео по хештегам, анализирует статистику",
            "icon": "📱",
            "category": "scraper",
            "config_preview": {
                "type": "scraper",
                "features": [
                    {"name": "Парсинг по хештегам", "description": "Сбор видео", "priority": "must"},
                    {"name": "Аналитика", "description": "Статистика просмотров", "priority": "should"}
                ],
                "database": "postgresql"
            }
        },
        {
            "id": "telegram_bot",
            "title": "🤖 Telegram Бот",
            "description": "AI-ассистент с админ-панелью",
            "icon": "🤖",
            "category": "bot",
            "config_preview": {
                "type": "bot",
                "features": [
                    {"name": "AI диалоги", "description": "Ответы на вопросы", "priority": "must"},
                    {"name": "Админ-панель", "description": "Управление", "priority": "should"}
                ],
                "database": "mongodb"
            }
        },
        {
            "id": "marketplace_api",
            "title": "🛒 Маркетплейс API",
            "description": "REST API для онлайн-магазина",
            "icon": "🛒",
            "category": "api",
            "config_preview": {
                "type": "api",
                "features": [
                    {"name": "Товары", "description": "CRUD операции", "priority": "must"},
                    {"name": "Корзина", "description": "Управление корзиной", "priority": "must"},
                    {"name": "Оплата", "description": "Интеграция платежей", "priority": "should"}
                ],
                "database": "postgresql",
                "authentication": True
            }
        },
        {
            "id": "finance_tracker",
            "title": "💰 Финансовый трекер",
            "description": "Учёт расходов с графиками",
            "icon": "💰",
            "category": "fullstack",
            "config_preview": {
                "type": "fullstack",
                "features": [
                    {"name": "Добавление транзакций", "description": "Расходы/доходы", "priority": "must"},
                    {"name": "Графики", "description": "Визуализация", "priority": "should"}
                ],
                "frontend": "react"
            }
        },
        {
            "id": "ai_content_generator",
            "title": "✨ Генератор контента",
            "description": "Создаёт посты для соцсетей",
            "icon": "✨",
            "category": "api",
            "config_preview": {
                "type": "api",
                "features": [
                    {"name": "Генерация текстов", "description": "На основе темы", "priority": "must"},
                    {"name": "Планирование", "description": "Отложенный постинг", "priority": "could"}
                ],
                "ai_settings": {"provider": "groq"}
            }
        },
        {
            "id": "url_shortener",
            "title": "🔗 Сокращатель ссылок",
            "description": "Как bit.ly с аналитикой",
            "icon": "🔗",
            "category": "api",
            "config_preview": {
                "type": "api",
                "features": [
                    {"name": "Сокращение URL", "description": "Генерация коротких ссылок", "priority": "must"},
                    {"name": "Статистика кликов", "description": "Аналитика переходов", "priority": "should"}
                ],
                "database": "redis"
            }
        },
        {
            "id": "chat_app",
            "title": "💬 Чат-приложение",
            "description": "Real-time чат с WebSocket",
            "icon": "💬",
            "category": "fullstack",
            "config_preview": {
                "type": "fullstack",
                "features": [
                    {"name": "Real-time сообщения", "description": "WebSocket", "priority": "must"},
                    {"name": "Комнаты", "description": "Групповые чаты", "priority": "should"}
                ],
                "frontend": "react"
            }
        },
        {
            "id": "blog_platform",
            "title": "📝 Платформа для блогов",
            "description": "Medium-клон с markdown",
            "icon": "📝",
            "category": "fullstack",
            "config_preview": {
                "type": "fullstack",
                "features": [
                    {"name": "Статьи", "description": "CRUD с markdown", "priority": "must"},
                    {"name": "Комментарии", "description": "Система комментов", "priority": "should"},
                    {"name": "Подписки", "description": "Follow авторов", "priority": "could"}
                ],
                "frontend": "vue"
            }
        }
    ]

    # Возвращаем 4 случайных примера
    selected = random.sample(all_examples, min(4, len(all_examples)))

    return {
        "examples": selected,
        "categories": list(set(e["category"] for e in all_examples)),
        "total_available": len(all_examples)
    }

# ============ ПРОЕКТЫ ============

@app.post("/projects")
async def create_project(request: CreateProjectRequest, background_tasks: BackgroundTasks):
    """Создание нового проекта"""
    try:
        # Сохраняем в базу
        project_id = await db.create_project(request.user_id, request.config.dict())

        # Запускаем генерацию в фоне
        background_tasks.add_task(build_project_task, project_id, request.config.dict())

        return {
            "success": True,
            "project_id": project_id,
            "status": "analyzing",
            "message": "Проект создан, начата генерация"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def build_project_task(project_id: str, config: dict):
    """Фоновая сборка проекта"""
    try:
        # Анализ и генерация
        await db.update_project(project_id, {"status": "generating"})
        result = await project_builder.analyze_and_build(config)

        # Сохраняем результат
        await db.update_project(project_id, {
            "status": "building",
            "files": result["files"],
            "architecture": result["architecture"]
        })

        # Деплой если нужно
        if config.get("auto_deploy"):
            await db.update_project(project_id, {"status": "deploying"})
            deploy_result = await deploy_engine.deploy(
                project_id,
                config["name"],
                result["files"]
            )

            await db.update_project(project_id, {
                "status": "live" if deploy_result["success"] else "error",
                "deploy_url": deploy_result.get("deploy_url"),
                "github_url": deploy_result.get("github_url"),
                "logs": deploy_result.get("error", "")
            })
        else:
            await db.update_project(project_id, {"status": "draft"})

    except Exception as e:
        await db.update_project(project_id, {
            "status": "error",
            "logs": str(e)
        })

@app.get("/projects/{project_id}")
async def get_project(project_id: str, user_id: str):
    """Получить проект"""
    project = await db.get_project(project_id, user_id)
    if not project:
        raise HTTPException(status_code=404, detail="Проект не найден")
    return project

@app.get("/projects")
async def list_projects(user_id: str):
    """Список проектов"""
    return {"projects": await db.list_projects(user_id)}

@app.delete("/projects/{project_id}")
async def delete_project(project_id: str, user_id: str):
    """Удалить проект"""
    await db.delete_project(project_id, user_id)
    return {"success": True}

# ============ СТАТУС ============

@app.get("/")
async def root():
    return {
        "status": "AI Developer Platform работает",
        "version": "2.0.0",
        "features": ["multi-ai", "dynamic-examples", "advanced-config"]
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
