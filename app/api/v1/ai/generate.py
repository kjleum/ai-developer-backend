from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from typing import List

from app.api.v1.deps import get_current_user, get_optional_user
from app.core.ai.manager import AIManager
from app.core.database import Database
from app.models.schemas import GenerateRequest

router = APIRouter()
ai_manager = AIManager()
db = Database()

@router.post("/generate")
async def generate_text(
    request: GenerateRequest,
    user: dict = Depends(get_current_user)
):
    response = await ai_manager.generate(
        prompt=request.prompt,
        provider=request.provider,
        model=request.model,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        json_mode=request.json_mode,
        user_id=user["id"]
    )

    await db.add_usage(
        user_id=user["id"],
        provider=request.provider or "auto",
        tokens=len(request.prompt.split()) + len(response.split()),
        cost=0.001,
        endpoint="/ai/generate"
    )

    return {"response": response}

@router.websocket("/stream")
async def stream_generate(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            prompt = data.get("prompt")
            provider = data.get("provider")
            user_id = data.get("user_id")

            async for chunk in ai_manager.stream_generate(prompt, provider, user_id=user_id):
                await websocket.send_text(chunk)
            await websocket.send_text("[DONE]")
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        await websocket.send_text(f"[ERROR] {str(e)}")
    finally:
        await websocket.close()

@router.get("/providers")
async def list_providers():
    return {"providers": ai_manager.get_available_providers()}
