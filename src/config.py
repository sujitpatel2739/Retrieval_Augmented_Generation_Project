from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    # OpenRouter API settings
    OPENAI_API_KEY: str = Field(..., env = "OPENAI_API_KEY")
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    
    # Weviate
    WEAVIATE_URL: str = "http://localhost:8080"
    WEAVIATE_PORT: int = 8080
    
    # LLM Settings
    router_model: str = "deepseek/deepseek-chat-v3-0324:free"
    reformulator_model: str = "deepseek/deepseek-chat-v3-0324:free"
    completion_model: str = "deepseek/deepseek-chat-v3-0324:free"
    answer_model: str = "deepseek/deepseek-chat-v3-0324:free"
    embedding_model: str = "all-MiniLM-L6-v2"
    
    # RAG Settings
    completion_threshold: float = 0.7
    
    JWT_SECRET_KEY: str
    ALGORITHM: str = 'HS256'
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60  # 1 hour
    
    DATABASE_URL:str = 'postgresql://JasonApex2739:@APEXJASONpg2739@localhost:5432/rag_main'
    
    class Config:
        env_file = ".env"
        validate_assignment = True
        
settings = Settings()