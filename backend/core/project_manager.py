
from supabase import create_client
import os
from typing import Dict, Any, List, Optional
from datetime import datetime

class ProjectManager:
    def __init__(self):
        self.client = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_KEY")
        )
        self.table = "projects"

    async def create(self, user_id: str, name: str, description: str, 
                     architecture: Dict, files: Dict[str, str], status: str = "created") -> str:
        """Создание нового проекта в базе"""
        data = {
            "user_id": user_id,
            "name": name,
            "description": description,
            "architecture": architecture,
            "files": files,
            "status": status,
            "url": None,
            "logs": "",
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }

        result = self.client.table(self.table).insert(data).execute()
        project_id = result.data[0]["id"]
        print(f"[DB] Проект создан: {project_id}")
        return project_id

    async def get(self, project_id: str, user_id: str) -> Optional[Dict]:
        """Получение проекта по ID"""
        result = self.client.table(self.table)\
            .select("*")\
            .eq("id", project_id)\
            .eq("user_id", user_id)\
            .execute()

        if result.data:
            return result.data[0]
        return None

    async def update_status(self, project_id: str, status: str, 
                           url: Optional[str] = None, logs: Optional[str] = None):
        """Обновление статуса проекта"""
        update_data = {
            "status": status,
            "updated_at": datetime.utcnow().isoformat()
        }
        if url:
            update_data["url"] = url
        if logs:
            update_data["logs"] = logs

        self.client.table(self.table)\
            .update(update_data)\
            .eq("id", project_id)\
            .execute()

        print(f"[DB] Статус обновлен: {project_id} -> {status}")

    async def update_files(self, project_id: str, files: Dict[str, str]):
        """Обновление файлов проекта"""
        self.client.table(self.table)\
            .update({
                "files": files,
                "updated_at": datetime.utcnow().isoformat()
            })\
            .eq("id", project_id)\
            .execute()

    async def list(self, user_id: str) -> List[Dict]:
        """Список всех проектов пользователя"""
        result = self.client.table(self.table)\
            .select("id, name, status, url, created_at, description")\
            .eq("user_id", user_id)\
            .order("created_at", desc=True)\
            .execute()
        return result.data

    async def delete(self, project_id: str, user_id: str):
        """Удаление проекта"""
        self.client.table(self.table)\
            .delete()\
            .eq("id", project_id)\
            .eq("user_id", user_id)\
            .execute()
