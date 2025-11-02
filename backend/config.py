"""
Configuration management for Jarvis.
Loads settings from .env file and provides typed access.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # OpenAI Configuration
    openai_api_key: str
    openai_model: str = "gpt-4o-mini"
    
    # Application Settings
    environment: str = "development"
    debug: bool = True
    log_level: str = "INFO"
    
    # Server Configuration
    backend_host: str = "0.0.0.0"
    backend_port: int = 9000
    mcp_server_host: str = "localhost"
    mcp_server_port: int = 8001
    
    # Database Paths
    chroma_db_path: str = "./data/chroma"
    calendar_db_path: str = "./data/calendar.json"
    docs_path: str = "./data/docs"
    
    # RAG Settings
    rag_chunk_size: int = 500
    rag_chunk_overlap: int = 50
    rag_top_k: int = 3
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    @property
    def mcp_server_url(self) -> str:
        """Full URL to MCP server."""
        return f"http://{self.mcp_server_host}:{self.mcp_server_port}"
    
    def ensure_directories(self) -> None:
        """Create required directories if they don't exist."""
        Path(self.chroma_db_path).mkdir(parents=True, exist_ok=True)
        Path(self.docs_path).mkdir(parents=True, exist_ok=True)
        Path(self.calendar_db_path).parent.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
