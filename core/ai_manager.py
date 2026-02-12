import os
import json
import asyncio
from typing import Dict, Any, Optional, List, Union
import aiohttp
import base64
from dataclasses import dataclass

@dataclass
class ModelCapability:
    chat: bool = True
    vision: bool = False
    code: bool = False
    embeddings: bool = False
    function_calling: bool = False

class AIManager:
    """Управление всеми AI API — облачные + локальные (Ollama)"""

    PROVIDERS = {
        # === ОСНОВНЫЕ БЕСПЛАТНЫЕ ОБЛАЧНЫЕ ===
        "groq": {
            "name": "Groq (Llama/Mixtral)",
            "models": ["llama-3.3-70b-versatile", "mixtral-8x7b-32768", "gemma2-9b-it", "deepseek-r1-distill-llama-70b"],
            "default_model": "llama-3.3-70b-versatile",
            "speed": "⚡ Мгновенно",
            "cost": "Бесплатно",
            "limits": "20/min, 600/day",
            "url": "https://console.groq.com/keys",
            "priority": 1,
            "capabilities": ModelCapability(chat=True, code=True, function_calling=True)
        },
        "gemini": {
            "name": "Google Gemini Pro",
            "models": ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.5-flash-8b"],
            "default_model": "gemini-1.5-flash",
            "speed": "🚀 Быстро",
            "cost": "Бесплатно",
            "limits": "60/min",
            "url": "https://makersuite.google.com/app/apikey",
            "priority": 2,
            "capabilities": ModelCapability(chat=True, vision=True, code=True)
        },
        "together": {
            "name": "Together AI",
            "models": ["meta-llama/Llama-3.3-70B-Instruct-Turbo", "mistralai/Mixtral-8x7B-Instruct-v0.1", "Qwen/Qwen2.5-72B-Instruct"],
            "default_model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "speed": "🚀 Быстро",
            "cost": "$5 стартовых",
            "limits": "Зависит от баланса",
            "url": "https://api.together.xyz/settings/api-keys",
            "priority": 3,
            "capabilities": ModelCapability(chat=True, code=True)
        },
        "deepseek": {
            "name": "DeepSeek",
            "models": ["deepseek-chat", "deepseek-coder", "deepseek-reasoner"],
            "default_model": "deepseek-coder",
            "speed": "🚀 Быстро",
            "cost": "1M токенов бесплатно",
            "limits": "500K input/day",
            "url": "https://platform.deepseek.com/api_keys",
            "priority": 4,
            "capabilities": ModelCapability(chat=True, code=True)
        },
        "mistral": {
            "name": "Mistral AI",
            "models": ["mistral-tiny", "mistral-small", "codestral-mamba", "mistral-medium"],
            "default_model": "mistral-small",
            "speed": "🚀 Быстро",
            "cost": "Бесплатный tier",
            "limits": "Зависит от tier",
            "url": "https://console.mistral.ai/api-keys",
            "priority": 5,
            "capabilities": ModelCapability(chat=True, code=True, function_calling=True)
        },
        "cohere": {
            "name": "Cohere",
            "models": ["command-r", "command-light", "command-r-plus"],
            "default_model": "command-r",
            "speed": "⚡ Быстро",
            "cost": "1000 запросов/мес",
            "limits": "Trial",
            "url": "https://dashboard.cohere.com/api-keys",
            "priority": 6,
            "capabilities": ModelCapability(chat=True, embeddings=True)
        },
        "ai21": {
            "name": "AI21 Labs",
            "models": ["jamba-1.5-mini", "jamba-1.5-large"],
            "default_model": "jamba-1.5-mini",
            "speed": "🚀 Быстро",
            "cost": "10K токенов/день",
            "limits": "Free tier",
            "url": "https://studio.ai21.com/account/api-key",
            "priority": 7,
            "capabilities": ModelCapability(chat=True)
        },
        "openrouter": {
            "name": "OpenRouter",
            "models": ["meta-llama/llama-3.3-70b-instruct", "anthropic/claude-3.5-sonnet", "google/gemini-2.0-flash-exp"],
            "default_model": "meta-llama/llama-3.3-70b-instruct",
            "speed": "🚀 Быстро",
            "cost": "Разные цены",
            "limits": "Много моделей",
            "url": "https://openrouter.ai/keys",
            "priority": 8,
            "capabilities": ModelCapability(chat=True, vision=True, code=True)
        },
        "huggingface": {
            "name": "HuggingFace",
            "models": ["meta-llama/Meta-Llama-3-8B-Instruct", "mistralai/Mistral-7B-Instruct-v0.2", "microsoft/Phi-3-mini-4k-instruct"],
            "default_model": "meta-llama/Meta-Llama-3-8B-Instruct",
            "speed": "🐢 Средне",
            "cost": "Бесплатно",
            "limits": "Rate limits",
            "url": "https://huggingface.co/settings/tokens",
            "priority": 9,
            "capabilities": ModelCapability(chat=True)
        },
        # === ЛОКАЛЬНЫЕ МОДЕЛИ (OLLAMA) ===
        "ollama": {
            "name": "Ollama (Локальные модели)",
            "models": ["llama3.3", "qwen2.5", "deepseek-coder-v2", "mistral", "codellama", "mixtral"],
            "default_model": "llama3.3",
            "speed": "⚡ Локально",
            "cost": "Бесплатно (GPU требуется)",
            "limits": "Без ограничений",
            "url": "http://localhost:11434",
            "priority": 0,  # Высший приоритет если доступен
            "capabilities": ModelCapability(chat=True, code=True),
            "local": True
        },
        # === СПЕЦИАЛИЗИРОВАННЫЕ ===
        "giga_embeddings": {
            "name": "Giga Embeddings (Сбер)",
            "models": ["Giga-Embeddings-v1"],
            "default_model": "Giga-Embeddings-v1",
            "speed": "🚀 Быстро",
            "cost": "Бесплатно",
            "limits": "API Сбера",
            "url": "https://developers.sber.ru/studio/",
            "priority": 10,
            "capabilities": ModelCapability(embeddings=True)
        },
        # === РЕЗЕРВ ===
        "mock": {
            "name": "Demo Mode (Offline)",
            "models": ["mock"],
            "default_model": "mock",
            "speed": "⚡ Мгновенно",
            "cost": "Бесплатно",
            "limits": "Без ограничений",
            "url": "",
            "priority": 99,
            "capabilities": ModelCapability(chat=True, code=True)
        }
    }

    def __init__(self):
        self.keys = {
            "groq": os.getenv("GROQ_API_KEY"),
            "gemini": os.getenv("GEMINI_API_KEY"),
            "together": os.getenv("TOGETHER_API_KEY"),
            "deepseek": os.getenv("DEEPSEEK_API_KEY"),
            "mistral": os.getenv("MISTRAL_API_KEY"),
            "cohere": os.getenv("COHERE_API_KEY"),
            "ai21": os.getenv("AI21_API_KEY"),
            "openrouter": os.getenv("OPENROUTER_API_KEY"),
            "huggingface": os.getenv("HF_API_KEY"),
            "openai": os.getenv("OPENAI_KEY"),
            "giga": os.getenv("GIGA_API_KEY")
        }
        self.ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.ollama_available = False

        # Проверяем Ollama при инициализации
        asyncio.create_task(self._check_ollama())

    async def _check_ollama(self):
        """Проверить доступность Ollama"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.ollama_url}/api/tags", timeout=aiohttp.ClientTimeout(total=2)):
                    self.ollama_available = True
                    print("✅ Ollama доступна локально")
        except:
            self.ollama_available = False

    def get_available_providers(self) -> List[Dict[str, Any]]:
        """Получить список доступных провайдеров с приоритетом"""
        available = []

        for key, config in self.PROVIDERS.items():
            if key == "ollama":
                has_key = self.ollama_available
            else:
                has_key = bool(self.keys.get(key)) or key == "mock"

            available.append({
                "id": key,
                **{k: v for k, v in config.items() if k != "capabilities"},
                "capabilities": {
                    "chat": config["capabilities"].chat,
                    "vision": config["capabilities"].vision,
                    "code": config["capabilities"].code,
                    "embeddings": config["capabilities"].embeddings,
                    "function_calling": config["capabilities"].function_calling
                },
                "available": has_key,
                "recommended": has_key and config["priority"] <= 3
            })

        # Сортируем: сначала локальные, потом по приоритету
        available.sort(key=lambda x: (0 if x.get("local") else 1, x["priority"]))
        return available

    def get_best_available_provider(self, capability: str = "chat") -> Optional[str]:
        """Вернуть лучший доступный провайдер с нужной способностью"""
        providers = self.get_available_providers()

        for p in providers:
            if not p["available"] or p["id"] == "mock":
                continue

            caps = p.get("capabilities", {})
            if capability == "chat" and caps.get("chat"):
                return p["id"]
            elif capability == "code" and caps.get("code"):
                return p["id"]
            elif capability == "embeddings" and caps.get("embeddings"):
                return p["id"]
            elif capability == "vision" and caps.get("vision"):
                return p["id"]

        return "mock"

    async def generate(self, prompt: str, provider: str = None, 
                      model: Optional[str] = None, 
                      temperature: float = 0.7,
                      max_tokens: int = 4000,
                      json_mode: bool = False,
                      stream: bool = False) -> Union[str, asyncio.AsyncGenerator]:
        """Генерация с fallback на другие провайдеры"""

        if provider is None or not self._is_available(provider):
            provider = self.get_best_available_provider("chat")

        providers_to_try = [provider] + [p["id"] for p in self.get_available_providers() 
                                         if p["id"] != provider and p["available"]]

        last_error = None
        for p in providers_to_try:
            try:
                if p == "ollama":
                    return await self._call_ollama(prompt, model, temperature, max_tokens, stream)
                elif p == "groq":
                    return await self._call_groq(prompt, model, temperature, max_tokens, json_mode)
                elif p == "gemini":
                    return await self._call_gemini(prompt, model, temperature, max_tokens)
                elif p == "together":
                    return await self._call_together(prompt, model, temperature, max_tokens)
                elif p == "deepseek":
                    return await self._call_deepseek(prompt, model, temperature, max_tokens)
                elif p == "mistral":
                    return await self._call_mistral(prompt, model, temperature, max_tokens)
                elif p == "cohere":
                    return await self._call_cohere(prompt, model, temperature, max_tokens)
                elif p == "ai21":
                    return await self._call_ai21(prompt, model, temperature, max_tokens)
                elif p == "openrouter":
                    return await self._call_openrouter(prompt, model, temperature, max_tokens)
                elif p == "huggingface":
                    return await self._call_huggingface(prompt, model, temperature, max_tokens)
                elif p == "openai":
                    return await self._call_openai(prompt, model, temperature, max_tokens)
                elif p == "mock":
                    return self._mock_response(prompt, json_mode)
            except Exception as e:
                print(f"❌ {p} failed: {str(e)[:100]}")
                last_error = e
                continue

        raise Exception(f"Все AI провайдеры недоступны. Последняя ошибка: {last_error}")

    async def generate_embeddings(self, texts: List[str], provider: str = None) -> List[List[float]]:
        """Генерация эмбеддингов для RAG"""
        if provider is None:
            provider = self.get_best_available_provider("embeddings")

        if provider == "cohere":
            return await self._call_cohere_embeddings(texts)
        elif provider == "giga_embeddings":
            return await self._call_giga_embeddings(texts)
        elif provider == "ollama":
            return await self._call_ollama_embeddings(texts)
        else:
            # Fallback на mock
            return [[0.0] * 768 for _ in texts]

    def _is_available(self, provider: str) -> bool:
        if provider == "mock":
            return True
        if provider == "ollama":
            return self.ollama_available
        return bool(self.keys.get(provider))

    # === API CALLS ===

    async def _call_ollama(self, prompt, model, temperature, max_tokens, stream=False):
        """Локальная генерация через Ollama"""
        model = model or self.PROVIDERS["ollama"]["default_model"]

        async with aiohttp.ClientSession() as session:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": stream,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }

            if stream:
                async with session.post(
                    f"{self.ollama_url}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=300)
                ) as resp:
                    async for line in resp.content:
                        if line:
                            data = json.loads(line)
                            if "response" in data:
                                yield data["response"]
            else:
                async with session.post(
                    f"{self.ollama_url}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=300)
                ) as resp:
                    if resp.status != 200:
                        raise Exception(f"Ollama error: {await resp.text()}")

                    full_response = ""
                    async for line in resp.content:
                        if line:
                            try:
                                data = json.loads(line)
                                if "response" in data:
                                    full_response += data["response"]
                            except:
                                pass
                    return full_response

    async def _call_ollama_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Генерация эмбеддингов через Ollama"""
        embeddings = []
        async with aiohttp.ClientSession() as session:
            for text in texts:
                payload = {
                    "model": "nomic-embed-text",  # Модель для эмбеддингов
                    "prompt": text
                }
                async with session.post(
                    f"{self.ollama_url}/api/embeddings",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    data = await resp.json()
                    embeddings.append(data.get("embedding", [0.0] * 768))
        return embeddings

    async def _call_groq(self, prompt, model, temperature, max_tokens, json_mode=False):
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
            if json_mode:
                payload["response_format"] = {"type": "json_object"}

            async with session.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=60)
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
                "generationConfig": {
                    "temperature": temperature, 
                    "maxOutputTokens": max_tokens
                }
            }
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                if resp.status != 200:
                    raise Exception(f"Gemini error: {await resp.text()}")
                data = await resp.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]

    async def _call_together(self, prompt, model, temperature, max_tokens):
        model = model or self.PROVIDERS["together"]["default_model"]
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {self.keys['together']}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            async with session.post(
                "https://api.together.xyz/v1/chat/completions",
                headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=60)
            ) as resp:
                if resp.status != 200:
                    raise Exception(f"Together error: {await resp.text()}")
                data = await resp.json()
                return data["choices"][0]["message"]["content"]

    async def _call_deepseek(self, prompt, model, temperature, max_tokens):
        model = model or self.PROVIDERS["deepseek"]["default_model"]
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {self.keys['deepseek']}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            async with session.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=60)
            ) as resp:
                if resp.status != 200:
                    raise Exception(f"DeepSeek error: {await resp.text()}")
                data = await resp.json()
                return data["choices"][0]["message"]["content"]

    async def _call_mistral(self, prompt, model, temperature, max_tokens):
        model = model or self.PROVIDERS["mistral"]["default_model"]
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {self.keys['mistral']}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            async with session.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=60)
            ) as resp:
                if resp.status != 200:
                    raise Exception(f"Mistral error: {await resp.text()}")
                data = await resp.json()
                return data["choices"][0]["message"]["content"]

    async def _call_cohere(self, prompt, model, temperature, max_tokens):
        model = model or self.PROVIDERS["cohere"]["default_model"]
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {self.keys['cohere']}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": model,
                "message": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            async with session.post(
                "https://api.cohere.ai/v1/chat",
                headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=60)
            ) as resp:
                if resp.status != 200:
                    raise Exception(f"Cohere error: {await resp.text()}")
                data = await resp.json()
                return data["text"]

    async def _call_cohere_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Cohere embeddings для RAG"""
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {self.keys['cohere']}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "embed-english-v3.0",
                "texts": texts,
                "input_type": "search_document"
            }
            async with session.post(
                "https://api.cohere.ai/v1/embed",
                headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                data = await resp.json()
                return data.get("embeddings", [])

    async def _call_giga_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Giga Embeddings от Сбера"""
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {self.keys['giga']}",
                "Content-Type": "application/json"
            }
            embeddings = []
            for text in texts:
                payload = {"text": text}
                async with session.post(
                    "https://gigachat.devices.sberbank.ru/api/v1/embeddings",
                    headers=headers, json=payload,
                    ssl=False  # Для тестовой среды
                ) as resp:
                    data = await resp.json()
                    embeddings.append(data.get("embedding", [0.0] * 768))
            return embeddings

    async def _call_ai21(self, prompt, model, temperature, max_tokens):
        model = model or self.PROVIDERS["ai21"]["default_model"]
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {self.keys['ai21']}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            async with session.post(
                "https://api.ai21.com/studio/v1/chat/completions",
                headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=60)
            ) as resp:
                if resp.status != 200:
                    raise Exception(f"AI21 error: {await resp.text()}")
                data = await resp.json()
                return data["choices"][0]["message"]["content"]

    async def _call_openrouter(self, prompt, model, temperature, max_tokens):
        model = model or self.PROVIDERS["openrouter"]["default_model"]
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {self.keys['openrouter']}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://ai-developer.app",
                "X-Title": "AI Developer Platform"
            }
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=60)
            ) as resp:
                if resp.status != 200:
                    raise Exception(f"OpenRouter error: {await resp.text()}")
                data = await resp.json()
                return data["choices"][0]["message"]["content"]

    async def _call_huggingface(self, prompt, model, temperature, max_tokens):
        model = model or self.PROVIDERS["huggingface"]["default_model"]
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {self.keys['huggingface']}",
                "Content-Type": "application/json"
            }
            payload = {
                "inputs": prompt,
                "parameters": {
                    "temperature": temperature,
                    "max_new_tokens": max_tokens,
                    "return_full_text": False
                }
            }
            async with session.post(
                f"https://api-inference.huggingface.co/models/{model}",
                headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=120)
            ) as resp:
                if resp.status != 200:
                    raise Exception(f"HF error: {await resp.text()}")
                data = await resp.json()
                if isinstance(data, list) and len(data) > 0:
                    return data[0].get("generated_text", "")
                return str(data)

    async def _call_openai(self, prompt, model, temperature, max_tokens):
        model = model or "gpt-3.5-turbo"
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
                headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=60)
            ) as resp:
                if resp.status != 200:
                    raise Exception(f"OpenAI error: {await resp.text()}")
                data = await resp.json()
                return data["choices"][0]["message"]["content"]

    def _mock_response(self, prompt, json_mode):
        """Demo режим без API"""
        if json_mode:
            return json.dumps({
                "type": "api",
                "name": "demo_project",
                "stack": ["python", "fastapi"],
                "files": ["main.py", "requirements.txt"],
                "description": "Demo project"
            })
        return "# Demo code\nprint('Hello from AI Developer Platform')"

    @staticmethod
    def clean_json_response(text: str) -> str:
        """Очистка JSON от markdown"""
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()
