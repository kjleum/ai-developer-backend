
from typing import Dict, Any
from core.ai_core import AIAgent

class CodeGenerator:
    def __init__(self):
        self.ai = AIAgent()

    async def generate(self, architecture: Dict[str, Any]) -> Dict[str, str]:
        """
        Генерация всех файлов проекта на основе архитектуры
        """
        print(f"[Generator] Создание проекта типа: {architecture['type']}")

        # Используем AI для генерации кода
        files = await self.ai.generate_code(architecture)

        # Добавляем специфичные файлы в зависимости от типа
        if architecture["type"] == "bot":
            files.update(self._bot_specific_files())
        elif architecture["type"] == "web":
            files.update(self._web_specific_files())
        elif architecture["type"] == "parser":
            files.update(self._parser_specific_files())

        return files

    def _bot_specific_files(self) -> Dict[str, str]:
        """Дополнительные файлы для ботов"""
        return {
            "config.py": """
import os
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("BOT_TOKEN", "")
ADMIN_ID = os.getenv("ADMIN_ID", "")
DATABASE_URL = os.getenv("DATABASE_URL", "")
""",
            "handlers.py": """
from telegram import Update
from telegram.ext import ContextTypes

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет! Я работаю.")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Доступные команды: /start, /help")
"""
        }

    def _web_specific_files(self) -> Dict[str, str]:
        """Дополнительные файлы для веб-приложений"""
        return {
            "static/style.css": """
body { font-family: Arial, sans-serif; margin: 40px; }
.container { max-width: 800px; margin: 0 auto; }
""",
            "templates/index.html": """
<!DOCTYPE html>
<html>
<head>
    <title>{{ title }}</title>
    <link rel="stylesheet" href="/static/style.css">
</head>
<body>
    <div class="container">
        <h1>{{ message }}</h1>
    </div>
</body>
</html>
"""
        }

    def _parser_specific_files(self) -> Dict[str, str]:
        """Дополнительные файлы для парсеров"""
        return {
            "scraper.py": """
import aiohttp
import asyncio
from typing import List, Dict

class Scraper:
    def __init__(self):
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *args):
        await self.session.close()

    async def fetch(self, url: str) -> str:
        async with self.session.get(url) as response:
            return await response.text()

    async def parse(self, html: str) -> List[Dict]:
        # Реализация парсинга
        return []
""",
            "scheduler.py": """
from apscheduler.schedulers.asyncio import AsyncIOScheduler

class TaskScheduler:
    def __init__(self):
        self.scheduler = AsyncIOScheduler()

    def add_job(self, func, interval_minutes: int = 5):
        self.scheduler.add_job(func, 'interval', minutes=interval_minutes)

    def start(self):
        self.scheduler.start()

    def stop(self):
        self.scheduler.shutdown()
"""
        }
