"""Local embedding providers (sentence-transformers / HuggingFace)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.embeddings.embeddings_factory import EmbeddingsFactory

__all__ = ["EmbeddingsFactory", "get_embeddings", "resolve_torch_device"]


def __getattr__(name: str):
    from src.embeddings import embeddings_factory as _factory

    mapping = {
        "EmbeddingsFactory": _factory.EmbeddingsFactory,
        "get_embeddings": _factory.get_embeddings,
        "resolve_torch_device": _factory.resolve_torch_device,
    }
    if name in mapping:
        return mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
