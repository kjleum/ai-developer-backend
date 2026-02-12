from supabase import create_client, Client
from typing import Optional, Dict, Any, List
import os
from datetime import datetime

class Database:
    def __init__(self):
        self.supabase: Client = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_KEY")
        )

    async def create_project(self, user_id: str, config: Dict[str, Any]) -> str:
        data = {
            "user_id": user_id,
            "config": config,
            "status": "draft",
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        result = self.supabase.table("projects").insert(data).execute()
        return result.data[0]["id"]

    async def update_project(self, project_id: str, updates: Dict[str, Any]):
        updates["updated_at"] = datetime.utcnow().isoformat()
        self.supabase.table("projects").update(updates).eq("id", project_id).execute()

    async def get_project(self, project_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        result = self.supabase.table("projects").select("*").eq("id", project_id).eq("user_id", user_id).execute()
        return result.data[0] if result.data else None

    async def list_projects(self, user_id: str) -> List[Dict[str, Any]]:
        result = self.supabase.table("projects").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
        return result.data

    async def delete_project(self, project_id: str, user_id: str):
        self.supabase.table("projects").delete().eq("id", project_id).eq("user_id", user_id).execute()
