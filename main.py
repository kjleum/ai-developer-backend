from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv
import random
import json
import asyncio
from typing import Optional, List

load_dotenv()

from models.schemas import *
from core.ai_manager import AIManager
from core.project_builder import ProjectBuilder
from core.deploy_engine import DeployEngine
from core.database import Database
from core.media_processor import MediaProcessor
from core.agent_system import AgentSystem
from core.rag_engine import RAGEngine
from core.nlp_interface import NLPInterface

# Глобальные объекты
ai_manager = AIManager()
project_builder = ProjectBuilder()
deploy_engine = DeployEngine()
db = Database()
media_processor = MediaProcessor()
agent_system = AgentSystem(ai_manager, None, db)
rag_engine = RAGEngine(ai_manager)
nlp_interface = NLPInterface(ai_manager, db)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Инициализация при старте"""
    print("🚀 AI Developer Platform v4.0 запущен")
    print("📦 Модули: AI, Media, Agents, RAG, NLP")

    # Проверяем сервисы
    services = await media_processor.check_services()
    print(f"   Медиа-сервисы: {sum(services.values())}/{len(services)} доступно")

    # Инициализация RAG
    await rag_engine.initialize()

    yield
    print("👋 Завершение работы")

app = FastAPI(
    title="AI Developer Platform v4.0",
    description="""
    Полноценная AI-платформа для разработки:
    - 🤖 LLM (9+ провайдеров + Ollama)
    - 🎨 Генерация изображений (SD, Kandinsky)
    - 🎬 Генерация видео (Wan, Kandinsky Video)
    - 🎤 TTS/STT (Coqui, Whisper)
    - 🧠 AI Агенты (5 типов + Flowise/Activepieces)
    - 📚 RAG (Chroma/Qdrant/pgvector)
    - 💬 NLP интерфейс к БД
    """,
    version="4.0.0",
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

# ============ AI ENDPOINTS ============

@app.get("/ai/providers")
async def get_ai_providers():
    """Получить список доступных AI провайдеров"""
    providers = ai_manager.get_available_providers()
    best = ai_manager.get_best_available_provider()
    return {
        "providers": providers,
        "recommended": best,
        "total_available": len([p for p in providers if p["available"]]),
        "total_free": len([p for p in providers if p["cost"] == "Бесплатно"]),
        "capabilities": ["chat", "vision", "code", "embeddings", "function_calling"]
    }

@app.post("/ai/generate")
async def ai_generate(request: GenerateRequest):
    """Генерация текста через AI"""
    try:
        response = await ai_manager.generate(
            prompt=request.prompt,
            provider=request.provider,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            json_mode=request.json_mode
        )
        return {"success": True, "response": response, "provider": request.provider or "auto"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ai/embeddings")
async def ai_embeddings(request: EmbeddingsRequest):
    """Генерация эмбеддингов для RAG"""
    try:
        embeddings = await ai_manager.generate_embeddings(
            texts=request.texts,
            provider=request.provider
        )
        return {"success": True, "embeddings": embeddings, "count": len(embeddings)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ai/stream")
async def ai_stream(websocket: WebSocket):
    """Стриминг генерации через WebSocket"""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            prompt = data.get("prompt")
            provider = data.get("provider")

            # Стриминг через Ollama или другой провайдер
            if provider == "ollama":
                async for chunk in ai_manager._call_ollama(prompt, stream=True):
                    await websocket.send_text(chunk)
            else:
                response = await ai_manager.generate(prompt, provider)
                await websocket.send_text(response)

            await websocket.send_text("[DONE]")
    except Exception as e:
        await websocket.send_text(f"[ERROR] {str(e)}")
    finally:
        await websocket.close()

# ============ MEDIA ENDPOINTS ============

@app.get("/media/services")
async def get_media_services():
    """Проверить доступность медиа-сервисов"""
    services = await media_processor.check_services()
    return {
        "services": services,
        "available": sum(services.values()),
        "total": len(services)
    }

@app.post("/media/image/generate")
async def generate_image(request: ImageGenerationRequest):
    """Генерация изображения"""
    try:
        result = await media_processor.generate_image(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            model=request.model,
            steps=request.steps,
            cfg_scale=request.cfg_scale
        )
        return {"success": True, **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/media/image/upscale")
async def upscale_image(request: UpscaleRequest):
    """Увеличение разрешения"""
    try:
        result = await media_processor.upscale(request.image, request.scale)
        return {"success": True, **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/media/video/generate")
async def generate_video(request: VideoGenerationRequest):
    """Генерация видео"""
    try:
        result = await media_processor.generate_video(
            prompt=request.prompt,
            image=request.image,
            duration=request.duration,
            model=request.model
        )
        return {"success": True, **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/media/audio/tts")
async def text_to_speech(request: TTSRequest):
    """Текст в речь"""
    try:
        result = await media_processor.text_to_speech(
            text=request.text,
            voice=request.voice,
            language=request.language,
            speed=request.speed
        )
        return {"success": True, **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/media/audio/stt")
async def speech_to_text(request: STTRequest):
    """Речь в текст"""
    try:
        result = await media_processor.speech_to_text(
            audio=request.audio,
            language=request.language,
            model=request.model
        )
        return {"success": True, **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/media/audio/clone-voice")
async def clone_voice(request: VoiceCloneRequest):
    """Клонирование голоса"""
    try:
        result = await media_processor.clone_voice(
            audio_samples=request.samples,
            name=request.name
        )
        return {"success": True, **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============ AGENT ENDPOINTS ============

@app.get("/agents")
async def get_agents():
    """Получить список агентов"""
    return {
        "agents": agent_system.get_agents(),
        "default_agents": ["developer", "analyst", "support", "manager", "creative"]
    }

@app.post("/agents/{agent_id}/run")
async def run_agent(agent_id: str, request: AgentRunRequest):
    """Запустить агента"""
    try:
        result = await agent_system.run_agent(
            agent_id=agent_id,
            user_input=request.input,
            context=request.context
        )
        return {"success": True, **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agents/create")
async def create_agent(request: CreateAgentRequest):
    """Создать нового агента"""
    try:
        agent = await agent_system.create_agent(
            name=request.name,
            description=request.description,
            capabilities=request.capabilities,
            system_prompt=request.system_prompt,
            tools=request.tools,
            custom_params=request.custom_params
        )
        return {"success": True, "agent": {"id": agent.id, "name": agent.name}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agents/tasks")
async def get_agent_tasks(status: Optional[str] = None):
    """Получить задачи агентов"""
    return {"tasks": agent_system.get_tasks(status)}

# ============ RAG ENDPOINTS ============

@app.post("/rag/collections")
async def create_collection(request: CreateCollectionRequest):
    """Создать коллекцию для RAG"""
    try:
        result = await rag_engine.create_collection(
            name=request.name,
            dimension=request.dimension,
            metadata=request.metadata
        )
        return {"success": True, **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag/collections/{collection}/documents")
async def add_documents(collection: str, request: AddDocumentsRequest):
    """Добавить документы в коллекцию"""
    try:
        from core.rag_engine import Document
        documents = [
            Document(
                id=doc.id or str(random.randint(10000, 99999)),
                content=doc.content,
                metadata=doc.metadata,
                embedding=doc.embedding
            )
            for doc in request.documents
        ]

        result = await rag_engine.add_documents(collection, documents)
        return {"success": True, **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag/collections/{collection}/search")
async def search_documents(collection: str, request: SearchRequest):
    """Поиск по коллекции"""
    try:
        results = await rag_engine.search(
            collection=collection,
            query=request.query,
            top_k=request.top_k,
            filter_metadata=request.filter
        )
        return {"success": True, "results": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag/chat")
async def rag_chat(request: RAGChatRequest):
    """Чат с документами"""
    try:
        result = await rag_engine.chat_with_documents(
            collection=request.collection,
            query=request.query,
            system_prompt=request.system_prompt,
            chat_history=request.history,
            top_k=request.top_k
        )
        return {"success": True, **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============ NLP INTERFACE ENDPOINTS ============

@app.post("/nlp/command")
async def nlp_command(request: NLPCommandRequest):
    """Выполнить команду на естественном языке"""
    try:
        # Парсим команду
        command = await nlp_interface.parse_command(request.command, request.context)

        # Выполняем если уверены
        if command.confidence > 0.7:
            result = await nlp_interface.execute_command(command, request.user_id)
            return {"success": True, "parsed": command.__dict__, "result": result}
        else:
            # Иначе чат-режим
            chat_result = await nlp_interface.chat_with_data(request.command, request.user_id)
            return {"success": True, "mode": "chat", **chat_result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/nlp/chat")
async def nlp_chat(request: NLPChatRequest):
    """Свободный чат с данными"""
    try:
        result = await nlp_interface.chat_with_data(request.message, request.user_id)
        return {"success": True, **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============ PROJECT ENDPOINTS (обновлённые) ============

TRENDING_TOPICS = [
    "Telegram бот для продаж", "CRM система", "Сервис бронирования",
    "API для доставки", "AI чат", "Финансовый трекер",
    "Маркетплейс", "Система лояльности", "Аналитика данных",
    "SaaS платформа", "ERP система", "Парсер товаров"
]

@app.get("/examples")
async def get_examples():
    """Генерировать динамические примеры проектов"""
    try:
        selected_topics = random.sample(TRENDING_TOPICS, min(4, len(TRENDING_TOPICS)))
        examples = []

        for topic in selected_topics:
            prompt = f"""Создай описание проекта "{topic}" для разработчика.
Ответь JSON: {{"title": "...", "description": "...", "type": "api/bot/saas/marketplace/crm/erp", "features": [], "stack": []}}"""

            try:
                response = await ai_manager.generate(prompt=prompt, temperature=0.8, max_tokens=500, json_mode=True)
                data = json.loads(ai_manager.clean_json_response(response))

                examples.append({
                    "id": f"example_{topic.replace(' ', '_').lower()}",
                    "title": data.get("title", topic),
                    "description": data.get("description", f"Проект: {topic}"),
                    "icon": get_icon_for_type(data.get("type", "api")),
                    "category": data.get("type", "api"),
                    "config_preview": {
                        "type": data.get("type", "api"),
                        "name": topic,
                        "features": [{"name": f, "description": f, "priority": "must"} for f in data.get("features", [])[:3]],
                        "database": "postgresql"
                    }
                })
            except:
                examples.append(create_fallback_example(topic))

        return {"examples": examples, "total_available": len(TRENDINGING_TOPICS)}
    except Exception as e:
        return {"examples": [create_fallback_example(t) for t in random.sample(TRENDING_TOPICS, 4)], "error": str(e)}

def get_icon_for_type(project_type: str) -> str:
    icons = {
        "api": "🔌", "bot": "🤖", "frontend": "🎨", "saas": "☁️",
        "marketplace": "🛒", "crm": "👥", "erp": "📊", "scraper": "🔍",
        "fullstack": "⚡", "cli": "⌨️"
    }
    return icons.get(project_type, "📦")

def create_fallback_example(topic: str) -> dict:
    return {
        "id": f"fallback_{topic.replace(' ', '_').lower()}",
        "title": f"📦 {topic}",
        "description": f"Полнофункциональный сервис для {topic.lower()}",
        "icon": "🚀",
        "category": "api",
        "config_preview": {"type": "api", "name": topic, "features": []}
    }

@app.post("/projects")
async def create_project(request: CreateProjectRequest, background_tasks: BackgroundTasks):
    """Создание нового проекта"""
    try:
        project_id = await db.create_project(request.user_id, request.config.dict())
        background_tasks.add_task(build_project_task, project_id, request.config.dict())

        return {
            "success": True,
            "project_id": project_id,
            "status": "analyzing",
            "message": "Проект создан, начата генерация"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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

async def build_project_task(project_id: str, config: dict):
    """Фоновая сборка проекта"""
    try:
        await db.update_project(project_id, {"status": "generating"})

        # Генерация
        result = await project_builder.analyze_and_build(config)

        await db.update_project(project_id, {
            "status": "building",
            "files": result["files"],
            "architecture": result["architecture"],
            "tech_stack": result["tech_stack"]
        })

        # Деплой если нужно
        if config.get("auto_deploy"):
            await db.update_project(project_id, {"status": "deploying"})
            deploy_result = await deploy_engine.deploy(project_id, config["name"], result["files"])

            await db.update_project(project_id, {
                "status": "live" if deploy_result["success"] else "error",
                "deploy_url": deploy_result.get("deploy_url"),
                "github_url": deploy_result.get("github_url")
            })
        else:
            await db.update_project(project_id, {"status": "draft"})

    except Exception as e:
        await db.update_project(project_id, {"status": "error", "logs": str(e)})

# ============ STATUS ============

@app.get("/")
async def root():
    """Статус платформы"""
    providers = ai_manager.get_available_providers()
    media_services = await media_processor.check_services()

    return {
        "status": "AI Developer Platform v4.0 работает",
        "version": "4.0.0",
        "modules": {
            "ai": {"providers": len([p for p in providers if p["available"]]), "status": "active"},
            "media": {"services": sum(media_services.values()), "status": "active"},
            "agents": {"count": len(agent_system.agents), "status": "active"},
            "rag": {"status": "active"},
            "nlp": {"status": "active"}
        },
        "features": [
            "multi-ai-providers", "local-ollama", "image-generation", 
            "video-generation", "tts-stt", "ai-agents", "rag-system",
            "nlp-database", "auto-deployment"
        ],
        "docs": "/docs"
    }

@app.get("/health")
async def health():
    """Health check"""
    return {"status": "ok", "version": "4.0.0"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
