"""Settings configuration using pydantic-settings."""
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings."""
    
    # LLM Configuration
    model_api_base: str = Field(
        default="http://localhost:11434/v1",
        description="OpenAI-compatible API base URL (Ollama default)"
    )
    model_api_key: str = Field(
        default="ollama",
        description="API key (Ollama uses 'ollama' as dummy key)"
    )
    model_name: str = Field(
        default="qwen2.5-coder:7b",
        description="Model name to use"
    )
    model_temperature: float = Field(default=0.1)
    
    # Embedding Configuration
    embedding_api_base: str = Field(
        default="http://localhost:11434/v1",
        description="Embedding API base URL"
    )
    embedding_api_key: str = Field(default="ollama")
    embedding_model: str = Field(default="nomic-embed-text")
    
    # ChromaDB Configuration
    chroma_persist_dir: str = Field(default="./data/chroma")
    chroma_collection_name: str = Field(default="rag_documents")
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_debug: bool = Field(default=True)
    
    # Document Ingestion
    docs_dir: str = Field(default="./data/docs")
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=200)
    
    # Eval
    eval_model: str = Field(default="qwen2.5-coder:7b")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


settings = Settings()