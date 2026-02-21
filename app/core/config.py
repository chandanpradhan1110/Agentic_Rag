from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Groq LLM
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"
    groq_temperature: float = 0.1
    groq_max_tokens: int = 1024

    # Embeddings & retrieval
    embedding_model: str = "all-MiniLM-L6-v2"
    top_k_chunks: int = 5
    chunk_size: int = 512
    chunk_overlap: int = 64
    max_rewrite_attempts: int = 2

    # Storage
    db_path: Path = Path("data/rag.db")
    upload_dir: Path = Path("data/uploads")
    faiss_dir: Path = Path("data/faiss")
    log_dir: Path = Path("data/logs")

    # Upload limits
    max_file_size_mb: int = 50
    allowed_extensions: set[str] = {".pdf", ".txt", ".docx", ".csv", ".md"}

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024

    def ensure_dirs(self):
        for d in [self.upload_dir, self.faiss_dir, self.log_dir]:
            d.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
