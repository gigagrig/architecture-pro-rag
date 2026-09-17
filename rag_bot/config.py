"""Environment configuration shared by the API and Telegram worker."""

from dataclasses import dataclass, field
import os
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent.parent


def load_environment() -> None:
    load_dotenv(ROOT / ".env", override=False)


def secret_from_env(name: str, default_file: str = "") -> str:
    value = os.getenv(name, "").strip()
    if value:
        return value
    filename = os.getenv(f"{name}_FILE", default_file)
    if not filename:
        return ""
    path = Path(filename)
    if not path.is_absolute():
        path = ROOT / path
    return path.read_text(encoding="utf-8").strip()


@dataclass(frozen=True)
class Settings:
    index_dir: Path
    cache_folder: str | None
    local_files_only: bool
    top_k: int
    min_score: float
    llm_base_url: str
    llm_model: str
    llm_api_key: str = field(repr=False)
    llm_timeout: float

    @classmethod
    def from_env(cls) -> "Settings":
        load_environment()
        settings = cls(
            index_dir=Path(os.getenv("INDEX_DIR", str(ROOT / "vector_index"))),
            cache_folder=os.getenv("EMBEDDING_CACHE") or None,
            local_files_only=os.getenv("LOCAL_FILES_ONLY", "false").lower() == "true",
            top_k=int(os.getenv("RAG_TOP_K", "5")),
            min_score=float(os.getenv("RAG_MIN_SCORE", "0.70")),
            llm_base_url=os.getenv("LLM_BASE_URL", "").rstrip("/"),
            llm_model=os.getenv("LLM_MODEL", ""),
            llm_api_key=secret_from_env("LLM_API_KEY"),
            llm_timeout=float(os.getenv("LLM_TIMEOUT", "120")),
        )
        if not 1 <= settings.top_k <= 10:
            raise ValueError("RAG_TOP_K must be between 1 and 10")
        if not -1 <= settings.min_score <= 1:
            raise ValueError("RAG_MIN_SCORE must be between -1 and 1")
        if settings.llm_timeout <= 0:
            raise ValueError("LLM_TIMEOUT must be positive")
        if not settings.llm_base_url or not settings.llm_model:
            raise ValueError("Set LLM_BASE_URL and LLM_MODEL in .env")
        return settings
