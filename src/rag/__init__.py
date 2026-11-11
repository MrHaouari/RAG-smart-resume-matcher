"""RAG orchestration: retrieve resume chunks + generate match analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.rag.ingestion import IngestionResult, ParentChildIngestion, sanitize_text
    from src.rag.pipeline import RAGPipeline, ResumeMatcherRAG

__all__ = [
    "IngestionResult",
    "ParentChildIngestion",
    "RAGPipeline",
    "ResumeMatcherRAG",
    "sanitize_text",
]


def __getattr__(name: str):
    from src.rag import ingestion as _ingestion
    from src.rag import pipeline as _pipeline

    mapping = {
        "IngestionResult": _ingestion.IngestionResult,
        "ParentChildIngestion": _ingestion.ParentChildIngestion,
        "sanitize_text": _ingestion.sanitize_text,
        "RAGPipeline": _pipeline.RAGPipeline,
        "ResumeMatcherRAG": _pipeline.ResumeMatcherRAG,
    }
    if name in mapping:
        return mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
