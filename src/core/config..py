from pydantic import BaseSettings


class Settings(BaseSettings):
    JWT_SECRET_KEY: str
    ALGORITHM: str = 'HS256'
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60  # 1 hour
    
    DATABASE_URL:str = 'postgresql://JasonApex2739:@APEXJASONpg2739@localhost:5432/rag_main'
    
    class Config:
        env_file = ".env"
        
settings = Settings()