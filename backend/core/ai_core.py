
import openai
import json
import os
from typing import Dict, Any

class AIAgent:
    def __init__(self):
        self.client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_KEY"))
        self.model = "gpt-4o"  # или gpt-4o-mini для экономии

    async def analyze_project(self, description: str) -> Dict[str, Any]:
        """
        AI анализирует описание проекта и выбирает архитектуру
        """
        prompt = f"""
        Проанализируй запрос пользователя и выбери оптимальную архитектуру.

        Запрос: "{description}"

        Определи:
        1. Тип проекта (bot, web, parser, api, marketplace, trading, content_machine)
        2. Стек технологий
        3. Необходимые модули
        4. Оценку сложности (1-10)
        5. Примерное время разработки

        Ответ строго в JSON формате:
        {{
            "type": "тип проекта",
            "stack": {{
                "backend": "Python/Node/Go",
                "frontend": "React/Vue/HTML/None",
                "database": "PostgreSQL/Mongo/Supabase/None",
                "additional": ["Redis", "Celery", etc]
            }},
            "modules": ["module1", "module2", "module3"],
            "complexity": 5,
            "estimated_hours": 10,
            "description": "краткое описание что будет сделано"
        }}
        """

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Ты - опытный технический архитектор. Отвечай только JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)
        print(f"[AI] Архитектура выбрана: {result['type']}, сложность: {result['complexity']}/10")
        return result

    async def generate_code(self, architecture: Dict[str, Any]) -> Dict[str, str]:
        """
        Генерация всех файлов проекта
        """
        files = {}

        # Генерируем каждый модуль
        for module in architecture["modules"]:
            code = await self._generate_module(module, architecture)
            files[f"{module}.py"] = code

        # Добавляем конфигурационные файлы
        files["requirements.txt"] = self._generate_requirements(architecture)
        files["Dockerfile"] = self._generate_dockerfile(architecture)
        files["README.md"] = self._generate_readme(architecture)

        return files

    async def _generate_module(self, module_name: str, architecture: Dict) -> str:
        """Генерация одного модуля"""
        prompt = f"""
        Напиши код для модуля "{module_name}".

        Тип проекта: {architecture['type']}
        Стек: {architecture['stack']['backend']}

        Требования:
        - Рабочий, production-ready код
        - Обработка ошибок
        - Комментарии на русском
        - Следование best practices

        Напиши полный код файла {module_name}.py
        """

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Ты - senior разработчик. Пиши чистый, рабочий код."},
                {"role": "user", "content": prompt}
            ]
        )

        return response.choices[0].message.content

    async def fix_code(self, current_files: Dict[str, str], architecture: Dict, feedback: str) -> Dict[str, str]:
        """
        Исправление кода по фидбеку пользователя
        """
        files_list = "\n".join([f"{k}:")
        prompt = f"""
        Текущий код проекта:
        {files_list}

        Запрос на изменение: "{feedback}"

        Архитектура проекта: {json.dumps(architecture, ensure_ascii=False)}

        Исправь код согласно запросу. Верни полный обновленный код всех файлов.
        Формат ответа:
        ФАЙЛ: имя_файла.py
        ```python
        код
        ```

        ФАЙЛ: следующий_файл.py
        ...
        """

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        # Парсим ответ и извлекаем файлы
        return self._parse_files_from_response(response.choices[0].message.content)

    def _parse_files_from_response(self, text: str) -> Dict[str, str]:
        """Извлечение файлов из ответа AI"""
        files = {}
        import re
        pattern = r'ФАЙЛ:\s*(.+?)\n```python\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        for filename, code in matches:
            files[filename.strip()] = code.strip()
        return files

    def _generate_requirements(self, arch: Dict) -> str:
        """Генерация requirements.txt"""
        base = ["fastapi", "uvicorn", "python-dotenv"]
        if arch["stack"]["database"] == "PostgreSQL":
            base.extend(["asyncpg", "sqlalchemy"])
        elif arch["stack"]["database"] == "Supabase":
            base.append("supabase")
        if "bot" in arch["type"]:
            base.append("python-telegram-bot")
        return "\n".join(base)

    def _generate_dockerfile(self, arch: Dict) -> str:
        return """FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
"""

    def _generate_readme(self, arch: Dict) -> str:
        return f"""# {arch['type']}

Автоматически сгенерировано AI Developer.

## Стек
- Backend: {arch['stack']['backend']}
- Frontend: {arch['stack']['frontend']}
- Database: {arch['stack']['database']}

## Запуск
```bash
pip install -r requirements.txt
python main.py
```
"""
