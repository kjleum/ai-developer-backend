from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List, Dict, Any, Literal
from enum import Enum
from datetime import datetime

# ----- AI -----
class GenerateRequest(BaseModel):
    prompt: str
    provider: Optional[str] = None
    model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4000
    json_mode: bool = False

class EmbeddingsRequest(BaseModel):
    texts: List[str]
    provider: Optional[str] = None

class ImageGenerationRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    model: str = "sd"
    steps: int = 30
    cfg_scale: float = 7.0

class VideoGenerationRequest(BaseModel):
    prompt: str
    duration: int = 5
    model: str = "wan"

class TTSRequest(BaseModel):
    text: str
    voice: str = "ru_RU"
    speed: float = 1.0

class STTRequest(BaseModel):
    audio: str

# ----- Проекты -----
class ProjectType(str, Enum):
    API = "api"
    BOT = "bot"
    FRONTEND = "frontend"
    FULLSTACK = "fullstack"
    SAAS = "saas"
    MARKETPLACE = "marketplace"
    CRM = "crm"
    ERP = "erp"
    SCRAPER = "scraper"
    CLI = "cli"
    MOBILE = "mobile"
    GAME = "game"
    ML = "ml"
    IOT = "iot"
    BLOCKCHAIN = "blockchain"

class ProjectFeature(BaseModel):
    name: str
    description: str
    priority: Literal["must", "should", "could"] = "must"

class ProjectConfig(BaseModel):
    name: str
    description: str
    type: ProjectType
    features: List[ProjectFeature] = []
    database: str = "postgresql"
    frontend_framework: Optional[str] = "react"
    backend_framework: Optional[str] = "fastapi"
    auto_deploy: bool = True
    infrastructure: bool = True
    tests: bool = True
    docs: bool = True
    cicd: bool = False
    monitoring: bool = False

# ----- Аутентификация -----
class UserRegister(BaseModel):
    email: EmailStr
    password: str
    full_name: Optional[str] = ""

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class UserOut(BaseModel):
    id: int
    email: str
    full_name: Optional[str]
    is_admin: bool
    balance: float
    settings: dict
    created_at: datetime

# ----- Ключи -----
class APIKeyIn(BaseModel):
    provider: str
    api_key: str

class APIKeyOut(BaseModel):
    provider: str
    created_at: datetime

# ----- Напоминания -----
class ReminderIn(BaseModel):
    title: str
    description: Optional[str] = ""
    remind_at: datetime

class ReminderOut(ReminderIn):
    id: int
    is_active: bool
    created_at: datetime
