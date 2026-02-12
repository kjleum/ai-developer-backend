from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from enum import Enum

class AIProvider(str, Enum):
    GROQ = "groq"
    GEMINI = "gemini"
    OPENAI = "openai"
    MOCK = "mock"

class ProjectType(str, Enum):
    API = "api"
    BOT = "bot"
    FRONTEND = "frontend"
    SCRAPER = "scraper"
    FULLSTACK = "fullstack"
    MOBILE = "mobile"
    CLI = "cli"
    LIBRARY = "library"

class DatabaseType(str, Enum):
    NONE = "none"
    POSTGRESQL = "postgresql"
    MONGODB = "mongodb"
    REDIS = "redis"
    SUPABASE = "supabase"

class FrontendFramework(str, Enum):
    NONE = "none"
    REACT = "react"
    VUE = "vue"
    VANILLA = "vanilla"
    TELEGRAM_MINI_APP = "telegram_mini_app"

class ProjectFeature(BaseModel):
    name: str
    description: str
    priority: Literal["must", "should", "could"] = "must"

class AISettings(BaseModel):
    provider: AIProvider = AIProvider.GROQ
    model: Optional[str] = None
    temperature: float = Field(0.7, ge=0.0, le=1.0)
    max_tokens: int = Field(4000, ge=500, le=8000)

class ProjectConfig(BaseModel):
    name: str
    description: str
    type: ProjectType
    features: List[ProjectFeature] = []
    database: DatabaseType = DatabaseType.NONE
    frontend: FrontendFramework = FrontendFramework.NONE
    ai_settings: AISettings = AISettings()
    auto_deploy: bool = True
    platform: Literal["render", "vercel", "none"] = "render"
    authentication: bool = False
    admin_panel: bool = False
    api_documentation: bool = True
    tests: bool = False
    docker: bool = False
    env_vars: Dict[str, str] = {}

class CreateProjectRequest(BaseModel):
    user_id: str
    config: ProjectConfig

class ProjectStatus(str, Enum):
    DRAFT = "draft"
    ANALYZING = "analyzing"
    GENERATING = "generating"
    BUILDING = "building"
    DEPLOYING = "deploying"
    LIVE = "live"
    ERROR = "error"

class ProjectResponse(BaseModel):
    id: str
    status: ProjectStatus
    config: ProjectConfig
    files: Dict[str, str] = {}
    url: Optional[str] = None
    logs: List[str] = []
    created_at: str
    updated_at: str
