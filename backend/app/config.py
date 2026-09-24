from pydantic import BaseSettings

class Settings(BaseSettings):
    # FastAPI settings
    title: str = "Codebase Onboarding Assistant"
    version: str = "1.0.0"
    
    # Database settings
    database_url: str
    
    # Redis settings
    redis_url: str
    
    # Claude API settings
    claude_api_key: str
    
    # Other settings
    webhook_secret: str

    class Config:
        env_file = ".env"

settings = Settings()