import json
from typing import Dict, Any, List
from .ai_manager import AIManager

class ProjectBuilder:
    def __init__(self):
        self.ai = AIManager()

    async def analyze_and_build(self, config: Dict[str, Any]) -> Dict[str, Any]:
        analysis = await self._analyze_project(config)
        architecture = await self._generate_architecture(config, analysis)
        files = await self._generate_all_files(config, architecture)

        if config.get("docker"):
            files["Dockerfile"] = await self._generate_dockerfile(config, architecture)
            files["docker-compose.yml"] = await self._generate_docker_compose(config, architecture)

        if config.get("tests"):
            files["tests/test_main.py"] = await self._generate_tests(config, architecture)

        return {
            "architecture": architecture,
            "files": files,
            "tech_stack": architecture.get("stack", []),
            "estimated_cost": self._estimate_cost(architecture)
        }

    async def _analyze_project(self, config: Dict[str, Any]) -> str:
        features_text = "\n".join([
            f"- {f['name']}: {f['description']} (приоритет: {f.get('priority', 'must')})"
            for f in config.get("features", [])
        ])

        prompt = f"""Проанализируй техническое задание.

Название: {config['name']}
Тип: {config['type']}
Описание: {config['description']}

Функции:
{features_text}

Требования:
- База данных: {config.get('database', 'none')}
- Frontend: {config.get('frontend', 'none')}
- Авторизация: {'да' if config.get('authentication') else 'нет'}

Определи сложность (1-10), время разработки, ключевые вызовы."""

        ai_config = config.get("ai_settings", {})
        return await self.ai.generate(
            prompt=prompt,
            provider=ai_config.get("provider", "groq"),
            model=ai_config.get("model"),
            temperature=0.5
        )

    async def _generate_architecture(self, config, analysis):
        features_json = json.dumps([f["name"] for f in config.get("features", [])], ensure_ascii=False)

        prompt = f"""Создай архитектуру проекта в JSON.

Название: {config['name']}
Тип: {config['type']}
Описание: {config['description']}
Функции: {features_json}

Ответь строго JSON с полями: type, name, description, stack, files, structure, endpoints, database_schema, env_vars"""

        ai_config = config.get("ai_settings", {})
        response = await self.ai.generate(
            prompt=prompt,
            provider=ai_config.get("provider", "groq"),
            model=ai_config.get("model"),
            temperature=0.3,
            json_mode=True
        )

        try:
            clean = self.ai.clean_json_response(response)
            return json.loads(clean)
        except:
            return {
                "type": config["type"],
                "name": config["name"],
                "description": config["description"],
                "stack": ["python", "fastapi"],
                "files": ["main.py", "requirements.txt"],
                "structure": {"main.py": "Основной файл"},
                "endpoints": [],
                "database_schema": "",
                "env_vars": []
            }

    async def _generate_all_files(self, config, architecture):
        files = {}
        ai_config = config.get("ai_settings", {})

        for filename in architecture.get("files", ["main.py"]):
            prompt = self._create_file_prompt(filename, config, architecture)
            content = await self.ai.generate(
                prompt=prompt,
                provider=ai_config.get("provider", "groq"),
                model=ai_config.get("model"),
                temperature=0.4
            )
            files[filename] = self._clean_code(content, filename)

        return files

    def _create_file_prompt(self, filename, config, architecture):
        file_type = filename.split(".")[-1]
        base = f"""Напиши код для файла `{filename}`.

Проект: {config['name']}
Тип: {architecture['type']}
Стек: {', '.join(architecture['stack'])}"""

        if file_type == "py":
            base += "\n- Type hints\n- Python 3.11+"
            if config.get("database") != "none":
                base += f"\n- БД: {config['database']}"
            if config.get("authentication"):
                base += "\n- JWT авторизация"

        base += f"\n\nСодержимое `{filename}`:"
        return base

    def _clean_code(self, content, filename):
        content = content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines)
        return content.strip()

    async def _generate_dockerfile(self, config, architecture):
        prompt = f"Создай Dockerfile для Python проекта. Стек: {', '.join(architecture['stack'])}"
        ai_config = config.get("ai_settings", {})
        return await self.ai.generate(prompt=prompt, provider=ai_config.get("provider", "groq"), temperature=0.2)

    async def _generate_docker_compose(self, config, architecture):
        services = ["app"]
        if config.get("database") == "postgresql":
            services.append("postgres")
        elif config.get("database") == "mongodb":
            services.append("mongo")
        prompt = f"Создай docker-compose.yml с сервисами: {', '.join(services)}"
        ai_config = config.get("ai_settings", {})
        return await self.ai.generate(prompt=prompt, provider=ai_config.get("provider", "groq"), temperature=0.2)

    async def _generate_tests(self, config, architecture):
        prompt = f"Создай pytest тесты для {config['name']}. Endpoints: {architecture.get('endpoints', [])}"
        ai_config = config.get("ai_settings", {})
        return await self.ai.generate(prompt=prompt, provider=ai_config.get("provider", "groq"), temperature=0.3)

    def _estimate_cost(self, architecture):
        files_count = len(architecture.get("files", []))
        return {
            "complexity": min(files_count * 2, 10),
            "estimated_hours": files_count * 4,
            "ai_calls": files_count + 2
        }
