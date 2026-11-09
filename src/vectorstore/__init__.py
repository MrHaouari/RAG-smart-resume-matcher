"""Persistent local vector store (ChromaDB)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.vectorstore.chromadb_manager import VectorStoreManager

__all__ = ["VectorStoreManager"]


def __getattr__(name: str):
    if name == "VectorStoreManager":
        from src.vectorstore.chromadb_manager import VectorStoreManager as _Manager

        return _Manager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
