from pydantic_settings import BaseSettings
from typing import Optional, List
import os


class Settings(BaseSettings):
    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_RELOAD: bool = True
    
    # OpenAI Settings
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    
    # Ollama Settings
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "deepseek-r1:8b"
    OLLAMA_EMBEDDING_MODEL: str = "nomic-embed-text"
    
    # RAG Settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 500
    TOP_K_RETRIEVAL: int = 5
    SIMILARITY_THRESHOLD: float = 0.7
    
    # File Paths
    DATA_DIR: str = "data"
    RAW_DATA_DIR: str = "data/raw"
    PROCESSED_DATA_DIR: str = "data/processed"
    EMBEDDINGS_DIR: str = "data/embeddings"
    
    # OCR Settings
    TESSERACT_CONFIG: str = "--psm 6"
    OCR_LANGUAGES: List[str] = ["ben", "eng"]
    
    # Vector Database Settings
    FAISS_INDEX_TYPE: str = "cosine"  # Options: l2, dot, cosine
    VECTOR_DIMENSION: int = 1536  # For OpenAI embeddings
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/app.log"
    
    # UI Settings
    STREAMLIT_PORT: int = 8501
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()