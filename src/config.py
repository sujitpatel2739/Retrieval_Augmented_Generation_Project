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
    
    SECRET_KEY: str = Field(..., env = "SECRET_KEY")
    ALGORITHM: str = 'HS256'
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60  # 1 hour
    
    DATABASE_URL:str = Field(..., env = "DATABASE_URL")
    DATABASE_HOST: str = 'localhost'
    DATABASE_PORT: str = '5432'
    DATABASE_USER: str = Field(..., env = "DATABASE_USER")
    DATABASE_PASSWORD: str = Field(..., env = "DATABASE_PASSWORD")
    DATABASE_NAME: str = 'rag_db'
    
    class Config:
        env_file = ".env"
        validate_assignment = True
        
settings = Settings()