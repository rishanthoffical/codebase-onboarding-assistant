from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings

BACKEND_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


class Settings(BaseSettings):
    # FastAPI settings
    title: str = "Codebase Onboarding Assistant"
    version: str = "1.0.0"

    # Database settings
    database_url: str = ""

    # Redis settings
    redis_url: str = "redis://localhost:6379/0"

    # Ollama settings
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "gemma3:latest"
    ollama_tool_model: str = "gemma4:26b"

    # Claude API settings
    claude_api_key: str = ""

    # Other settings
    webhook_secret: str = ""

    # Authentication
    auth_enabled: bool = False
    auth_secret: str = ""

    # Chroma vector database
    chroma_persist_directory: str = "../chroma_db"
    chroma_host: str = ""
    chroma_port: int = 8000

    # Cache and logging
    chat_cache_ttl: int = 3600
    log_level: str = "INFO"

    # Rate limiting
    chat_rate_limit: str = "30/minute"
    webhook_rate_limit: str = "10/minute"

    # Repository upload limits
    max_files: int = Field(default=5000, gt=0)
    max_upload_size_mb: int = Field(default=500, gt=0)
    max_single_file_mb: int = Field(default=10, gt=0)

    class Config:
        env_file = BACKEND_ENV_FILE
        case_sensitive = False


settings = Settings()
