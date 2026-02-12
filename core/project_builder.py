import json
import os
from typing import Dict, Any, List, Optional
from core.ai_manager import AIManager

class ProjectBuilder:
    """Продвинутый генератор проектов — от простых API до SaaS платформ"""

    def __init__(self):
        self.ai = AIManager()
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict[str, Any]:
        """Загрузить шаблоны проектов"""
        return {
            "api": {
                "stack": ["python", "fastapi", "postgresql"],
                "files": ["main.py", "models.py", "database.py", "schemas.py", "auth.py", "config.py", "requirements.txt", "Dockerfile", "docker-compose.yml"],
                "structure": "layered"
            },
            "bot": {
                "stack": ["python", "aiogram", "redis"],
                "files": ["bot.py", "handlers.py", "middlewares.py", "database.py", "config.py", "requirements.txt", "Dockerfile"],
                "structure": "modular"
            },
            "frontend": {
                "stack": ["react", "typescript", "tailwind", "vite"],
                "files": ["src/App.tsx", "src/main.tsx", "src/index.css", "index.html", "package.json", "tsconfig.json", "vite.config.ts", "Dockerfile"],
                "structure": "component"
            },
            "fullstack": {
                "stack": ["react", "fastapi", "postgresql", "docker"],
                "files": [
                    "frontend/src/App.tsx", "frontend/package.json",
                    "backend/main.py", "backend/models.py", "backend/requirements.txt",
                    "docker-compose.yml", "nginx.conf"
                ],
                "structure": "microservices"
            },
            "saas": {
                "stack": ["nextjs", "prisma", "postgresql", "stripe", "clerk"],
                "files": [
                    "app/layout.tsx", "app/page.tsx", "app/api/webhooks/route.ts",
                    "lib/prisma.ts", "lib/stripe.ts", "lib/auth.ts",
                    "components/Pricing.tsx", "components/Dashboard.tsx",
                    "package.json", "prisma/schema.prisma", ".env.example"
                ],
                "structure": "saas"
            },
            "marketplace": {
                "stack": ["nextjs", "postgresql", "redis", "elasticsearch", "docker"],
                "files": [
                    "app/layout.tsx", "app/page.tsx", "app/search/page.tsx",
                    "app/api/products/route.ts", "app/api/orders/route.ts",
                    "components/ProductCard.tsx", "components/SearchFilters.tsx",
                    "lib/db.ts", "lib/search.ts", "lib/cache.ts",
                    "package.json", "docker-compose.yml"
                ],
                "structure": "marketplace"
            }
        }

    async def analyze_and_build(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Полный цикл создания проекта любой сложности"""

        # 1. Глубокий анализ требований
        analysis = await self._deep_analysis(config)

        # 2. Архитектура системы
        architecture = await self._design_architecture(config, analysis)

        # 3. Выбор технологий
        tech_stack = await self._select_tech_stack(config, architecture)

        # 4. Генерация всех файлов
        files = await self._generate_all_files(config, architecture, tech_stack)

        # 5. Дополнительные компоненты
        if config.get("infrastructure"):
            files.update(await self._generate_infrastructure(config, tech_stack))

        if config.get("tests"):
            files.update(await self._generate_tests(config, architecture))

        if config.get("docs"):
            files.update(await self._generate_documentation(config, architecture))

        # 6. CI/CD
        if config.get("cicd"):
            files.update(await self._generate_cicd(config, tech_stack))

        return {
            "architecture": architecture,
            "tech_stack": tech_stack,
            "files": files,
            "analysis": analysis,
            "estimated_complexity": self._calculate_complexity(architecture),
            "deployment_ready": config.get("auto_deploy", False)
        }

    async def _deep_analysis(self, config: Dict) -> Dict[str, Any]:
        """Глубокий анализ требований с AI"""

        features_text = "\n".join([
            f"- {f['name']}: {f['description']} (приоритет: {f.get('priority', 'must')})"
            for f in config.get("features", [])
        ])

        prompt = f"""Проведи глубокий анализ требований для проекта enterprise-уровня.

НАЗВАНИЕ: {config['name']}
ТИП: {config['type']}
ОПИСАНИЕ: {config['description']}

ФУНКЦИОНАЛЬНЫЕ ТРЕБОВАНИЯ:
{features_text}

Определи архитектурный стиль, паттерны, риски, этапы разработки.
"""

        ai_config = config.get("ai_settings", {})
        analysis_text = await self.ai.generate(
            prompt=prompt,
            provider=ai_config.get("provider"),
            temperature=0.5,
            max_tokens=3000
        )

        return {
            "text": analysis_text,
            "features_count": len(config.get("features", [])),
            "complexity_score": min(len(config.get("features", [])) * 2, 10)
        }

    async def _design_architecture(self, config: Dict, analysis: Dict) -> Dict[str, Any]:
        """Проектирование архитектуры"""

        project_type = config["type"]
        template = self.templates.get(project_type, self.templates["api"])

        features = [f["name"] for f in config.get("features", [])]

        prompt = f"""Спроектируй архитектуру системы в формате JSON.

ПРОЕКТ: {config['name']}
ТИП: {project_type}
ФУНКЦИИ: {json.dumps(features, ensure_ascii=False)}

Ответь JSON с полями: project_type, architecture_style, layers, data_flow, databases, security, files_structure."""

        ai_config = config.get("ai_settings", {})
        response = await self.ai.generate(
            prompt=prompt,
            provider=ai_config.get("provider"),
            temperature=0.3,
            max_tokens=4000
        )

        try:
            clean = self.ai.clean_json_response(response)
            arch = json.loads(clean)
        except:
            arch = self._fallback_architecture(config, template)

        return arch

    def _fallback_architecture(self, config: Dict, template: Dict) -> Dict:
        """Резервная архитектура"""
        return {
            "project_type": config["type"],
            "architecture_style": "layered",
            "layers": [
                {"name": "api", "components": ["routes", "controllers"], "tech": "fastapi"},
                {"name": "business", "components": ["services", "validators"], "tech": "python"},
                {"name": "data", "components": ["repositories", "models"], "tech": "sqlalchemy"}
            ],
            "files_structure": {f: f"Module {f}" for f in template["files"]},
            "tech_stack": template["stack"]
        }

    async def _select_tech_stack(self, config: Dict, architecture: Dict) -> Dict[str, Any]:
        """Выбор технологического стека"""

        type_mapping = {
            "api": {"backend": "fastapi", "db": "postgresql", "cache": "redis"},
            "bot": {"backend": "aiogram", "db": "sqlite", "cache": "memory"},
            "frontend": {"ui": "react", "styling": "tailwind", "state": "zustand"},
            "fullstack": {"backend": "fastapi", "frontend": "react", "db": "postgresql"},
            "saas": {"framework": "nextjs", "auth": "clerk", "payments": "stripe", "db": "prisma"}
        }

        return type_mapping.get(config["type"], type_mapping["api"])

    async def _generate_all_files(self, config: Dict, architecture: Dict, tech_stack: Dict) -> Dict[str, str]:
        """Генерация всех файлов проекта"""
        files = {}

        file_structure = architecture.get("files_structure", {})
        ai_config = config.get("ai_settings", {})

        for filepath in file_structure.keys():
            try:
                content = await self._generate_file_content(
                    filepath, config, architecture, tech_stack, ai_config
                )
                files[filepath] = content
            except Exception as e:
                print(f"Ошибка генерации {filepath}: {e}")
                files[filepath] = self._fallback_content(filepath, config)

        return files

    async def _generate_file_content(self, filepath: str, config: Dict, architecture: Dict, tech_stack: Dict, ai_config: Dict) -> str:
        """Генерация содержимого конкретного файла"""

        ext = filepath.split(".")[-1]

        context = f"""Файл: {filepath}
Проект: {config['name']}
Тип: {config['type']}
Стек: {json.dumps(tech_stack, ensure_ascii=False)}
"""

        prompt = f"""Напиши production-ready код для файла `{filepath}`.

{context}

Требования:
- Код должен быть рабочим
- Type hints / strict types
- Обработка ошибок
- Докстринги

Выведи ТОЛЬКО код:
```
"""

        content = await self.ai.generate(
            prompt=prompt,
            provider=ai_config.get("provider"),
            temperature=0.3,
            max_tokens=6000
        )

        return self._clean_code(content, ext)

    def _clean_code(self, content: str, ext: str) -> str:
        """Очистка сгенерированного кода"""
        content = content.strip()

        if content.startswith("```"):
            lines = content.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines)

        return content.strip()

    def _fallback_content(self, filepath: str, config: Dict) -> str:
        """Резервное содержимое"""
        ext = filepath.split(".")[-1]

        fallbacks = {
            "py": f'# {filepath}\nprint("Hello from {config["name"]}")',
            "ts": f'console.log("{config["name"]}")',
            "tsx": f'export default function Component() {{ return <div>{config["name"]}</div> }}',
            "json": '{"name": "' + config["name"] + '"}',
            "yml": "version: '3.8'\nservices:",
            "Dockerfile": f"FROM python:3.11-slim\nWORKDIR /app\nCOPY . .\nCMD ['python', 'main.py']"
        }

        return fallbacks.get(ext, f"# {filepath}")

    async def _generate_infrastructure(self, config: Dict, tech_stack: Dict) -> Dict[str, str]:
        """Генерация инфраструктуры"""
        files = {}

        # Docker Compose
        compose_services = """version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/app
    depends_on:
      - db
    volumes:
      - ./:/app

  db:
    image: postgres:15-alpine
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
      POSTGRES_DB: app
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
"""
        files["docker-compose.yml"] = compose_services

        # Dockerfile
        dockerfile = """FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
"""
        files["Dockerfile"] = dockerfile

        return files

    async def _generate_tests(self, config: Dict, architecture: Dict) -> Dict[str, str]:
        """Генерация тестов"""
        return {
            "pytest.ini": "[pytest]\nasyncio_mode = auto\n",
            "tests/test_main.py": f"import pytest\n\nclass Test{config['name'].replace(' ', '')}:\n    async def test_health(self):\n        assert True\n"
        }

    async def _generate_documentation(self, config: Dict, architecture: Dict) -> Dict[str, str]:
        """Генерация документации"""
        return {
            "README.md": f"# {config['name']}\n\n{config['description']}\n\n## Установка\n\n```bash\npip install -r requirements.txt\n```",
            ".env.example": f"DATABASE_URL=postgresql://user:pass@localhost/{config['name'].lower().replace(' ', '_')}\nSECRET_KEY=change-me\n"
        }

    async def _generate_cicd(self, config: Dict, tech_stack: Dict) -> Dict[str, str]:
        """Генерация CI/CD"""
        return {
            ".github/workflows/ci.yml": f"""name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: pytest
"""
        }

    def _calculate_complexity(self, architecture: Dict) -> Dict[str, int]:
        """Расчёт сложности"""
        files_count = len(architecture.get("files_structure", {}))
        return {
            "files_count": files_count,
            "complexity_score": min(files_count * 2, 100),
            "estimated_hours": files_count * 2
        }
