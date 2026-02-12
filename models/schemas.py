from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from enum import Enum

# ============ AI ============

class AIProvider(str, Enum):
    GROQ = "groq"
    GEMINI = "gemini"
    TOGETHER = "together"
    DEEPSEEK = "deepseek"
    MISTRAL = "mistral"
    COHERE = "cohere"
    AI21 = "ai21"
    OPENROUTER = "openrouter"
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"
    OLLAMA = "ollama"
    MOCK = "mock"

class GenerateRequest(BaseModel):
    prompt: str
    provider: Optional[str] = None
    model: Optional[str] = None
    temperature: float = Field(0.7, ge=0.0, le=1.0)
    max_tokens: int = Field(4000, ge=100, le=8000)
    json_mode: bool = False

class EmbeddingsRequest(BaseModel):
    texts: List[str]
    provider: Optional[str] = None

# ============ MEDIA ============

class ImageGenerationRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    model: str = "sd"  # sd, kandinsky
    steps: int = 30
    cfg_scale: float = 7.0
    seed: int = -1

class UpscaleRequest(BaseModel):
    image: str  # base64
    scale: int = 2

class VideoGenerationRequest(BaseModel):
    prompt: str
    image: Optional[str] = None  # base64 для I2V
    duration: int = 4
    fps: int = 8
    model: str = "wan"  # wan, kandinsky

class TTSRequest(BaseModel):
    text: str
    voice: str = "default"
    language: str = "ru"
    speed: float = 1.0
    emotion: str = "neutral"

class STTRequest(BaseModel):
    audio: str  # base64
    language: str = "ru"
    model: str = "whisper"  # whisper, vosk

class VoiceCloneRequest(BaseModel):
    samples: List[str]  # base64 audio samples
    name: str

# ============ AGENTS ============

class AgentRunRequest(BaseModel):
    input: str
    context: Optional[Dict[str, Any]] = None
    stream: bool = False

class CreateAgentRequest(BaseModel):
    name: str
    description: str
    capabilities: List[str]
    system_prompt: str
    tools: List[str]
    custom_params: Optional[Dict[str, Any]] = None

# ============ RAG ============

class CreateCollectionRequest(BaseModel):
    name: str
    dimension: int = 768
    metadata: Optional[Dict[str, Any]] = None

class DocumentInput(BaseModel):
    id: Optional[str] = None
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None

class AddDocumentsRequest(BaseModel):
    documents: List[DocumentInput]

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    filter: Optional[Dict[str, Any]] = None

class RAGChatRequest(BaseModel):
    collection: str
    query: str
    system_prompt: Optional[str] = None
    history: Optional[List[Dict[str, str]]] = None
    top_k: int = 5

# ============ NLP ============

class NLPCommandRequest(BaseModel):
    command: str
    context: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None

class NLPChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = None

# ============ PROJECTS ============

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

class DatabaseType(str, Enum):
    NONE = "none"
    POSTGRESQL = "postgresql"
    MONGODB = "mongodb"
    REDIS = "redis"
    SUPABASE = "supabase"

class ProjectFeature(BaseModel):
    name: str
    description: str
    priority: Literal["must", "should", "could"] = "must"

class AISettings(BaseModel):
    provider: str = "groq"
    model: Optional[str] = None
    temperature: float = Field(0.7, ge=0.0, le=1.0)
    max_tokens: int = Field(4000, ge=500, le=8000)

class ProjectConfig(BaseModel):
    name: str
    description: str
    type: ProjectType
    features: List[ProjectFeature] = []
    database: DatabaseType = DatabaseType.POSTGRESQL
    frontend: Optional[str] = None
    ai_settings: AISettings = AISettings()
    auto_deploy: bool = True
    infrastructure: bool = True
    tests: bool = True
    docs: bool = True
    cicd: bool = False
    scalability: str = "medium"  # low, medium, high
    security: str = "standard"  # basic, standard, high
    env_vars: Dict[str, str] = {}

class CreateProjectRequest(BaseModel):
    user_id: str
    config: ProjectConfig
