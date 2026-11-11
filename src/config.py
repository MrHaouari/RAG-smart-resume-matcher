"""Central configuration for the Smart Resume Matcher pipeline.

All knobs used by parsing, embeddings, Chroma and Ollama live here so that
the rest of the codebase never hard-codes paths or model names.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESUMES_DIR = DATA_DIR / "resumes"
JOBS_DIR = DATA_DIR / "jobs"


@dataclass(frozen=True)
class Settings:
    """Immutable runtime settings (100% local, no cloud APIs)."""

    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2"
    OLLAMA_TEMPERATURE: float = 0.1

    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    CROSS_ENCODER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    INITIAL_RETRIEVAL_K: int = 15
    FINAL_RERANK_K: int = 4

    CHROMA_PERSIST_DIR: str = "./chroma_db"
    CHROMA_COLLECTION_NAME: str = "resume_matcher"

    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 80

    # Paramètres Parent-Document Retriever
    PARENT_CHUNK_SIZE: int = 900
    PARENT_CHUNK_OVERLAP: int = 100
    CHILD_CHUNK_SIZE: int = 180
    CHILD_CHUNK_OVERLAP: int = 30

    @property
    def chroma_persist_path(self) -> Path:
        """Resolve ``CHROMA_PERSIST_DIR`` against the project root.

        Relative paths stay stable even when Streamlit or the CLI is launched
        from another working directory.
        """
        path = Path(self.CHROMA_PERSIST_DIR)
        if path.is_absolute():
            return path
        return (PROJECT_ROOT / path).resolve()


settings = Settings()

# Module-level aliases (keep imports like ``from src.config import CHUNK_SIZE``).
OLLAMA_BASE_URL = settings.OLLAMA_BASE_URL
OLLAMA_MODEL = settings.OLLAMA_MODEL
OLLAMA_TEMPERATURE = settings.OLLAMA_TEMPERATURE
EMBEDDING_MODEL_NAME = settings.EMBEDDING_MODEL_NAME
CROSS_ENCODER_MODEL_NAME = settings.CROSS_ENCODER_MODEL_NAME
INITIAL_RETRIEVAL_K = settings.INITIAL_RETRIEVAL_K
FINAL_RERANK_K = settings.FINAL_RERANK_K
CHROMA_PERSIST_DIR = settings.CHROMA_PERSIST_DIR
CHROMA_COLLECTION_NAME = settings.CHROMA_COLLECTION_NAME
CHUNK_SIZE = settings.CHUNK_SIZE
CHUNK_OVERLAP = settings.CHUNK_OVERLAP
PARENT_CHUNK_SIZE = settings.PARENT_CHUNK_SIZE
PARENT_CHUNK_OVERLAP = settings.PARENT_CHUNK_OVERLAP
CHILD_CHUNK_SIZE = settings.CHILD_CHUNK_SIZE
CHILD_CHUNK_OVERLAP = settings.CHILD_CHUNK_OVERLAP
CHROMA_DIR = settings.chroma_persist_path
