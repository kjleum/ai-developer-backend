import os
import aiohttp
from typing import Dict, Any
from github import Github

class DeployEngine:
    def __init__(self):
        self.github_token = os.getenv("GITHUB_TOKEN")
        self.render_token = os.getenv("RENDER_API_KEY")
        self.github = Github(self.github_token) if self.github_token else None

    async def deploy(self, project_id: str, name: str, files: Dict[str, str]) -> Dict[str, Any]:
        try:
            repo = await self._create_repo(name)
            await self._push_files(repo["full_name"], files)
            service = await self._create_render_service(name, repo["clone_url"])

            return {
                "success": True,
                "github_url": repo["url"],
                "deploy_url": service["url"],
                "status": "deploying"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _create_repo(self, name: str) -> Dict[str, str]:
        if not self.github:
            raise Exception("GitHub token not configured")

        user = self.github.get_user()
        try:
            repo = user.create_repo(name, private=True, auto_init=True)
        except:
            repo = self.github.get_repo(f"{user.login}/{name}")

        return {
            "name": repo.name,
            "full_name": repo.full_name,
            "url": repo.html_url,
            "clone_url": repo.clone_url
        }

    async def _push_files(self, repo_full_name: str, files: Dict[str, str]):
        repo = self.github.get_repo(repo_full_name)
        for path, content in files.items():
            try:
                existing = repo.get_contents(path)
                repo.update_file(path, f"Update {path}", content, existing.sha)
            except:
                repo.create_file(path, f"Create {path}", content)

    async def _create_render_service(self, name: str, repo_url: str) -> Dict[str, str]:
        if not self.render_token:
            raise Exception("Render token not configured")

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
                "envVars": [{"key": "PYTHON_VERSION", "value": "3.11"}]
            }

            async with session.post(
                "https://api.render.com/v1/services",
                headers=headers, json=payload
            ) as resp:
                if resp.status not in [200, 201]:
                    raise Exception(f"Render error: {await resp.text()}")
                data = await resp.json()
                return {
                    "id": data["id"],
                    "name": data["name"],
                    "url": data["serviceDetails"]["url"]
                }
