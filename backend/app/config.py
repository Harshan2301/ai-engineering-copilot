from pydantic_settings import BaseSettings
from functools import lru_cache
from pathlib import Path


class Settings(BaseSettings):
    # Gemini
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-1.5-pro"
    GEMINI_EMBED_MODEL: str = "models/text-embedding-004"

    # ChromaDB
    CHROMA_PERSIST_DIR: str = "./chroma_data"
    CHROMA_COLLECTION_NAME: str = "engineering_docs"

    # File uploads
    UPLOAD_DIR: str = "./uploads"
    MAX_FILE_SIZE_MB: int = 50
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100

    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    ALLOWED_ORIGINS: str = "http://localhost:8501,http://frontend:8501"

    # Auth
    SECRET_KEY: str = "changeme-super-secret-key-32chars!!"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # Logging
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @property
    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",")]

    def ensure_dirs(self):
        Path(self.CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
        Path(self.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    return Settings()
