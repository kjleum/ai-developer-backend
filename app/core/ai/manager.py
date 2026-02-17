import os
import time
from typing import Optional, Dict, Any, List, AsyncGenerator

import httpx

from app.core.database import Database

db = Database()

OPENAI_COMPAT: Dict[str, Dict[str, str]] = {
    "openai":      {"base": "https://api.openai.com/v1", "key_env": "OPENAI_API_KEY"},
    "groq":        {"base": "https://api.groq.com/openai/v1", "key_env": "GROQ_API_KEY"},
    "together":    {"base": "https://api.together.xyz/v1", "key_env": "TOGETHER_API_KEY"},
    "mistral":     {"base": "https://api.mistral.ai/v1", "key_env": "MISTRAL_API_KEY"},
    "openrouter":  {"base": "https://openrouter.ai/api/v1", "key_env": "OPENROUTER_API_KEY"},
    "deepseek":    {"base": "https://api.deepseek.com/v1", "key_env": "DEEPSEEK_API_KEY"},
    "perplexity":  {"base": "https://api.perplexity.ai", "key_env": "PERPLEXITY_API_KEY"},
    "fireworks":   {"base": "https://api.fireworks.ai/inference/v1", "key_env": "FIREWORKS_API_KEY"},
    "xai":         {"base": "https://api.x.ai/v1", "key_env": "XAI_API_KEY"},
    # Custom OpenAI-compatible endpoint (self-hosted / proxy)
    "custom":      {"base": os.getenv("CUSTOM_OAI_BASE", "http://localhost:8080/v1"), "key_env": "CUSTOM_OAI_KEY"},
}

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "groq": "llama-3.1-70b-versatile",
    "together": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "mistral": "mistral-large-latest",
    "openrouter": "openai/gpt-4o-mini",
    "deepseek": "deepseek-chat",
    "perplexity": "sonar",
    "fireworks": "accounts/fireworks/models/llama-v3p1-70b-instruct",
    "xai": "grok-beta",
    "custom": os.getenv("CUSTOM_OAI_MODEL", "gpt-4o-mini"),
}

MODEL_PRESETS = {
    "openai": ["gpt-4o", "gpt-4o-mini", "o1-mini", "o1"],
    "groq": ["llama-3.1-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
    "mistral": ["mistral-large-latest", "mistral-small-latest", "codestral-latest"],
    "together": ["meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", "Qwen/Qwen2.5-72B-Instruct-Turbo"],
    "openrouter": ["openai/gpt-4o-mini", "anthropic/claude-3.5-sonnet", "google/gemini-1.5-pro"],
    "deepseek": ["deepseek-chat", "deepseek-reasoner"],
    "perplexity": ["sonar", "sonar-pro"],
    "fireworks": ["accounts/fireworks/models/llama-v3p1-70b-instruct"],
    "xai": ["grok-beta"],
    "custom": [os.getenv("CUSTOM_OAI_MODEL", "gpt-4o-mini")],
}

class AIManager:
    async def _get_key(self, provider: str, user_id: Optional[int]) -> Optional[str]:
        if user_id:
            k = await db.get_api_key(user_id, provider)
            if k:
                return k
        env = OPENAI_COMPAT.get(provider, {}).get("key_env")
        if env:
            return os.getenv(env)
        return None

    async def list_models(self, user_id: int) -> Dict[str, Any]:
        providers = []
        for p in OPENAI_COMPAT.keys():
            if await self._get_key(p, user_id):
                providers.append(p)
        return {
            "providers": providers,
            "models": {p: MODEL_PRESETS.get(p, []) for p in providers},
            "note": "Можно указывать любой model строкой — если провайдер поддерживает."
        }

    async def _pick_provider(self, user_id: int) -> str:
        # prefer providers with user keys
        for p in OPENAI_COMPAT.keys():
            if await self._get_key(p, user_id):
                return p
        raise ValueError("Нет доступных провайдеров: добавьте ключ в разделе 'Ключи'.")

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
        if not user_id:
            raise ValueError("user_id required")

        provider = provider or "auto"
        if provider == "auto":
            provider = await self._pick_provider(user_id)

        key = await self._get_key(provider, user_id)
        if not key:
            raise ValueError(f"Нет ключа для провайдера '{provider}'")

        base = OPENAI_COMPAT[provider]["base"].rstrip("/")
        model = model or DEFAULT_MODELS.get(provider, "gpt-4o-mini")

        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        if provider == "openrouter":
            headers["HTTP-Referer"] = os.getenv("OPENROUTER_REFERER", "https://localhost")
            headers["X-Title"] = os.getenv("OPENROUTER_TITLE", "AI Platform")

        payload: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        t0 = time.time()
        status = "ok"
        try:
            async with httpx.AsyncClient(timeout=90) as client:
                r = await client.post(f"{base}/chat/completions", headers=headers, json=payload)
                if r.status_code != 200:
                    status = f"err_{r.status_code}"
                    raise RuntimeError(r.text)
                data = r.json()
                text = data["choices"][0]["message"]["content"]
                return text
        finally:
            latency_ms = int((time.time() - t0) * 1000)
            # usage log is handled by route; still mark last_used
            await db.touch_api_key_last_used(user_id, provider)
            # project-level log can be extended later

    async def stream_generate(self, *args, **kwargs) -> AsyncGenerator[str, None]:
        # Simple fallback: no real streaming; yields once.
        text = await self.generate(*args, **kwargs)
        yield text
