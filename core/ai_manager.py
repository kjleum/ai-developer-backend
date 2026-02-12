import os
import json
import random
from typing import Dict, Any, Optional, List
import aiohttp

class AIManager:
    PROVIDERS = {
        "groq": {
            "name": "Groq",
            "models": ["mixtral-8x7b-32768", "llama2-70b-4096", "gemma-7b-it"],
            "default_model": "mixtral-8x7b-32768",
            "speed": "⚡ Мгновенно",
            "cost": "Бесплатно",
            "limits": "20/min, 600/day"
        },
        "gemini": {
            "name": "Google Gemini",
            "models": ["gemini-pro", "gemini-pro-vision"],
            "default_model": "gemini-pro",
            "speed": "🚀 Быстро",
            "cost": "Бесплатно",
            "limits": "60/min"
        },
        "openai": {
            "name": "OpenAI",
            "models": ["gpt-4", "gpt-3.5-turbo"],
            "default_model": "gpt-3.5-turbo",
            "speed": "🚀 Быстро",
            "cost": "Платно",
            "limits": "Зависит от баланса"
        },
        "mock": {
            "name": "Demo Mode",
            "models": ["mock"],
            "default_model": "mock",
            "speed": "⚡ Мгновенно",
            "cost": "Бесплатно",
            "limits": "Без ограничений"
        }
    }

    def __init__(self):
        self.keys = {
            "groq": os.getenv("GROQ_API_KEY"),
            "gemini": os.getenv("GEMINI_API_KEY"),
            "openai": os.getenv("OPENAI_KEY")
        }

    def get_available_providers(self) -> List[Dict[str, Any]]:
        available = []
        for key, config in self.PROVIDERS.items():
            has_key = bool(self.keys.get(key)) or key == "mock"
            available.append({
                "id": key,
                **config,
                "available": has_key,
                "recommended": key == "groq" and has_key
            })
        return available

    def get_models(self, provider: str) -> List[str]:
        return self.PROVIDERS.get(provider, {}).get("models", ["default"])

    async def generate(self, prompt: str, provider: str = "groq", 
                      model: Optional[str] = None, 
                      temperature: float = 0.7,
                      max_tokens: int = 4000,
                      json_mode: bool = False) -> str:
        providers_to_try = [provider] + [p for p in ["groq", "gemini", "openai", "mock"] if p != provider]

        last_error = None
        for p in providers_to_try:
            if not self._is_available(p):
                continue
            try:
                if p == "groq":
                    return await self._call_groq(prompt, model, temperature, max_tokens)
                elif p == "gemini":
                    return await self._call_gemini(prompt, model, temperature, max_tokens)
                elif p == "openai":
                    return await self._call_openai(prompt, model, temperature, max_tokens)
                elif p == "mock":
                    return self._mock_response(prompt, json_mode)
            except Exception as e:
                last_error = e
                continue

        raise Exception(f"All AI providers failed: {last_error}")

    def _is_available(self, provider: str) -> bool:
        if provider == "mock":
            return True
        return bool(self.keys.get(provider))

    async def _call_groq(self, prompt, model, temperature, max_tokens):
        model = model or self.PROVIDERS["groq"]["default_model"]
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {self.keys['groq']}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            async with session.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status != 200:
                    raise Exception(f"Groq error: {await resp.text()}")
                data = await resp.json()
                return data["choices"][0]["message"]["content"]

    async def _call_gemini(self, prompt, model, temperature, max_tokens):
        model = model or self.PROVIDERS["gemini"]["default_model"]
        async with aiohttp.ClientSession() as session:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.keys['gemini']}"
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens}
            }
            async with session.post(url, json=payload) as resp:
                if resp.status != 200:
                    raise Exception(f"Gemini error: {await resp.text()}")
                data = await resp.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]

    async def _call_openai(self, prompt, model, temperature, max_tokens):
        model = model or self.PROVIDERS["openai"]["default_model"]
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {self.keys['openai']}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            async with session.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers, json=payload
            ) as resp:
                if resp.status != 200:
                    raise Exception(f"OpenAI error: {await resp.text()}")
                data = await resp.json()
                return data["choices"][0]["message"]["content"]

    def _mock_response(self, prompt, json_mode):
        if "architecture" in prompt.lower() or "проанализируй" in prompt.lower():
            if json_mode:
                return json.dumps({
                    "type": "api",
                    "name": "demo_api",
                    "stack": ["python", "fastapi", "postgresql"],
                    "files": ["main.py", "models.py", "database.py", "requirements.txt"],
                    "description": "REST API с авторизацией"
                })
            return "Demo API проект на FastAPI"
        return "# Demo code\nprint('Hello from AI Developer')"

    @staticmethod
    def clean_json_response(text: str) -> str:
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()
