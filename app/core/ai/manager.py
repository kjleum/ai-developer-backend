import os
import json
import asyncio
import aiohttp
from typing import Optional, List, Dict, Any, AsyncGenerator
from app.core.database import Database

db = Database()

class AIManager:
    def __init__(self):
        self.global_keys = {
            "groq": os.getenv("GROQ_API_KEY"),
            "gemini": os.getenv("GEMINI_API_KEY"),
            "openai": os.getenv("OPENAI_API_KEY"),
            "together": os.getenv("TOGETHER_API_KEY"),
            "deepseek": os.getenv("DEEPSEEK_API_KEY"),
            "mistral": os.getenv("MISTRAL_API_KEY"),
            "cohere": os.getenv("COHERE_API_KEY"),
            "ai21": os.getenv("AI21_API_KEY"),
            "openrouter": os.getenv("OPENROUTER_API_KEY"),
            "hf": os.getenv("HF_API_KEY"),
        }
        self.default_models = {
            "openai": "gpt-4",
            "groq": "mixtral-8x7b-32768",
            "gemini": "gemini-pro",
            "together": "togethercomputer/llama-2-70b-chat",
            "deepseek": "deepseek-chat",
            "mistral": "mistral-large-latest",
            "cohere": "command",
            "ai21": "j2-ultra",
            "openrouter": "openai/gpt-3.5-turbo",
            "hf": "HuggingFaceH4/zephyr-7b-beta",
        }

    async def _get_key(self, provider: str, user_id: Optional[int] = None) -> Optional[str]:
        if user_id:
            user_key = await db.get_api_key(user_id, provider)
            if user_key:
                return user_key
        return self.global_keys.get(provider)

    async def generate(
        self,
        prompt: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        json_mode: bool = False,
        user_id: Optional[int] = None
    ) -> str:
        if not provider:
            for prov in self.global_keys:
                if await self._get_key(prov, user_id):
                    provider = prov
                    break
            if not provider:
                raise ValueError("Нет доступных AI-провайдеров")

        api_key = await self._get_key(provider, user_id)
        if not api_key:
            raise ValueError(f"Нет API ключа для провайдера {provider}")

        model = model or self.default_models.get(provider, "gpt-3.5-turbo")

        if provider == "openai":
            return await self._call_openai(prompt, model, temperature, max_tokens, api_key, json_mode)
        elif provider == "groq":
            return await self._call_groq(prompt, model, temperature, max_tokens, api_key, json_mode)
        else:
            raise ValueError(f"Провайдер {provider} не поддерживается")

    async def stream_generate(
        self,
        prompt: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        user_id: Optional[int] = None
    ) -> AsyncGenerator[str, None]:
        provider = provider or "openai"
        api_key = await self._get_key(provider, user_id)
        if not api_key:
            yield f"[ERROR] No API key for {provider}"
            return

        model = model or self.default_models.get(provider, "gpt-3.5-turbo")

        if provider == "openai":
            async for chunk in self._stream_openai(prompt, model, temperature, max_tokens, api_key):
                yield chunk
        else:
            yield f"[ERROR] Streaming not supported for {provider}"

    async def _call_openai(self, prompt, model, temperature, max_tokens, api_key, json_mode):
        import openai
        client = openai.AsyncOpenAI(api_key=api_key)
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"} if json_mode else None
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"OpenAI error: {e}")

    async def _call_groq(self, prompt, model, temperature, max_tokens, api_key, json_mode):
        async with aiohttp.ClientSession() as session:
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "response_format": {"type": "json_object"} if json_mode else None
            }
            async with session.post("https://api.groq.com/openai/v1/chat/completions",
                                    headers=headers, json=payload) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"Groq error {resp.status}: {text}")
                data = await resp.json()
                return data["choices"][0]["message"]["content"]

    async def _stream_openai(self, prompt, model, temperature, max_tokens, api_key):
        import openai
        client = openai.AsyncOpenAI(api_key=api_key)
        stream = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        )
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def get_available_providers(self) -> List[str]:
        return [p for p, key in self.global_keys.items() if key] + ["mock"]
