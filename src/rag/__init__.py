"""RAG orchestration: retrieve resume chunks + generate match analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.rag.pipeline import RAGPipeline, ResumeMatcherRAG

__all__ = ["RAGPipeline", "ResumeMatcherRAG"]


def __getattr__(name: str):
    from src.rag import pipeline as _pipeline

    mapping = {
        "RAGPipeline": _pipeline.RAGPipeline,
        "ResumeMatcherRAG": _pipeline.ResumeMatcherRAG,
    }
    if name in mapping:
        return mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
