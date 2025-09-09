from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    # OpenRouter API settings
    openai_api_key: str = Field(..., env = "OPENAI_API_KEY")
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    
    # Weviate
    weaviate_url: str = "http://localhost:8080"
    weaviate_port: int = 8080
    weaviate_primary_class: str = "temporary"
    
    # LLM Settings
    router_model: str = "deepseek/deepseek-chat-v3-0324:free"
    reformulator_model: str = "deepseek/deepseek-chat-v3-0324:free"
    completion_model: str = "deepseek/deepseek-chat-v3-0324:free"
    answer_model: str = "deepseek/deepseek-chat-v3-0324:free"
    embedding_model: str = "all-MiniLM-L6-v2"
    
    # RAG Settings
    completion_threshold: float = 0.7
    
    class Config:
        env_file = ".env"
        validate_assignment = True