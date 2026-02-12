
import os
import base64
import aiohttp
from typing import Dict, Any
from github import Github
import git
import tempfile
import shutil

class DeployEngine:
    def __init__(self):
        self.github_token = os.getenv("GITHUB_TOKEN")
        self.render_token = os.getenv("RENDER_API_KEY")  # rnd_hHa4drVJjMS4pekL4ovTphrV6yZz
        self.github = Github(self.github_token)
        self.render_api = "https://api.render.com/v1"

    async def deploy(self, project_id: str, files: Dict[str, str]) -> Dict[str, Any]:
        """
        Полный цикл деплоя:
        1. Создать GitHub репозиторий
        2. Залить код
        3. Создать сервис на Render
        4. Вернуть URL
        """
        try:
            print(f"[Deploy] Начало деплоя проекта {project_id}")

            # 1. Создаем репозиторий на GitHub
            repo_name = f"ai-project-{project_id[:8]}"
            repo = await self._create_github_repo(repo_name)
            print(f"[Deploy] GitHub репозиторий создан: {repo['url']}")

            # 2. Заливаем код
            await self._push_to_github(repo['full_name'], files)
            print(f"[Deploy] Код загружен на GitHub")

            # 3. Создаем сервис на Render
            render_service = await self._create_render_service(repo_name, repo['clone_url'])
            print(f"[Deploy] Render сервис создан: {render_service['url']}")

            return {
                "success": True,
                "url": render_service['url'],
                "github_url": repo['url'],
                "logs": f"Деплой успешен. Сервис: {render_service['name']}"
            }

        except Exception as e:
            print(f"[Deploy] Ошибка: {str(e)}")
            return {
                "success": False,
                "url": None,
                "logs": str(e)
            }

    async def redeploy(self, project_id: str, files: Dict[str, str]) -> Dict[str, Any]:
        """Передеплой существующего проекта"""
        # Находим существующий репозиторий
        repo_name = f"ai-project-{project_id[:8]}"

        try:
            # Обновляем код
            await self._push_to_github(f"your-username/{repo_name}", files)

            # Render автоматически перезапустится при новом коммите
            # Находим URL сервиса
            async with aiohttp.ClientSession() as session:
                headers = {"Authorization": f"Bearer {self.render_token}"}
                async with session.get(
                    f"{self.render_api}/services",
                    headers=headers
                ) as resp:
                    services = await resp.json()
                    for service in services:
                        if service['name'] == repo_name:
                            return {
                                "success": True,
                                "url": service['serviceDetails']['url'],
                                "logs": "Обновление запущено"
                            }

            return {"success": False, "url": None, "logs": "Сервис не найден"}

        except Exception as e:
            return {"success": False, "url": None, "logs": str(e)}

    async def _create_github_repo(self, name: str) -> Dict[str, str]:
        """Создание приватного репозитория на GitHub"""
        try:
            user = self.github.get_user()
            repo = user.create_repo(
                name=name,
                private=True,
                description=f"AI-generated project {name}",
                auto_init=True
            )
            return {
                "name": repo.name,
                "full_name": repo.full_name,
                "url": repo.html_url,
                "clone_url": repo.clone_url
            }
        except Exception as e:
            # Если репозиторий уже существует
            if "already exists" in str(e):
                repo = self.github.get_repo(f"{user.login}/{name}")
                return {
                    "name": repo.name,
                    "full_name": repo.full_name,
                    "url": repo.html_url,
                    "clone_url": repo.clone_url
                }
            raise e

    async def _push_to_github(self, repo_full_name: str, files: Dict[str, str]):
        """Загрузка файлов в репозиторий через GitHub API"""
        repo = self.github.get_repo(repo_full_name)

        for filename, content in files.items():
            try:
                # Пробуем получить существующий файл
                try:
                    existing = repo.get_contents(filename)
                    # Обновляем существующий
                    repo.update_file(
                        path=filename,
                        message=f"Update {filename}",
                        content=content,
                        sha=existing.sha
                    )
                except:
                    # Создаем новый
                    repo.create_file(
                        path=filename,
                        message=f"Create {filename}",
                        content=content
                    )
            except Exception as e:
                print(f"[Deploy] Ошибка загрузки {filename}: {e}")

    async def _create_render_service(self, name: str, repo_url: str) -> Dict[str, str]:
        """Создание веб-сервиса на Render"""
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {self.render_token}",
                "Content-Type": "application/json"
            }

            payload = {
                "type": "web_service",
                "name": name,
                "repo": repo_url,
                "branch": "main",
                "buildCommand": "pip install -r requirements.txt",
                "startCommand": "uvicorn main:app --host 0.0.0.0 --port $PORT",
                "envVars": [
                    {"key": "PYTHON_VERSION", "value": "3.11"}
                ]
            }

            async with session.post(
                f"{self.render_api}/services",
                headers=headers,
                json=payload
            ) as resp:
                if resp.status in [200, 201]:
                    data = await resp.json()
                    return {
                        "name": data['name'],
                        "url": data['serviceDetails']['url'],
                        "id": data['id']
                    }
                else:
                    error = await resp.text()
                    raise Exception(f"Render API error: {error}")

    async def get_logs(self, service_id: str) -> str:
        """Получение логов сервиса"""
        async with aiohttp.ClientSession() as session:
            headers = {"Authorization": f"Bearer {self.render_token}"}
            async with session.get(
                f"{self.render_api}/services/{service_id}/logs",
                headers=headers
            ) as resp:
                if resp.status == 200:
                    return await resp.text()
                return "Логи недоступны"
