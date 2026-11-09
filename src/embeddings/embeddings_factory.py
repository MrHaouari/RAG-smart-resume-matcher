"""Local HuggingFace / sentence-transformers embedding factory.

Embeddings are computed on-device (Apple Silicon MPS when available, else
CPU). No remote inference API is used; the model weights are loaded from
the local HuggingFace cache after the first download.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from langchain_huggingface import HuggingFaceEmbeddings

from src.config import settings

logger = logging.getLogger(__name__)


def resolve_torch_device() -> str:
    """Pick the best local device: MPS (Apple Silicon) > CUDA > CPU."""
    try:
        import torch

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        logger.warning("PyTorch is not installed; falling back to CPU embeddings.")
    return "cpu"


class EmbeddingsFactory:
    """Build a local ``HuggingFaceEmbeddings`` instance from project settings."""

    @staticmethod
    def create(
        model_name: str | None = None,
        device: str | None = None,
        normalize_embeddings: bool = True,
        **model_kwargs: Any,
    ) -> HuggingFaceEmbeddings:
        """Return a ready-to-use local embedding model.

        Args:
            model_name: HuggingFace model id. Defaults to
                ``settings.EMBEDDING_MODEL_NAME``.
            device: ``"mps"``, ``"cpu"`` or ``"cuda"``. Auto-detected if omitted.
            normalize_embeddings: L2-normalize vectors (recommended for cosine
                similarity in Chroma).
            **model_kwargs: Extra kwargs forwarded to ``SentenceTransformer``.

        Returns:
            A ``HuggingFaceEmbeddings`` bound to the local sentence-transformers
            runtime.
        """
        resolved_name = model_name or settings.EMBEDDING_MODEL_NAME
        resolved_device = device or resolve_torch_device()
        encode_kwargs = {"normalize_embeddings": normalize_embeddings}

        logger.info(
            "Loading local embeddings '%s' on device='%s' (normalize=%s).",
            resolved_name,
            resolved_device,
            normalize_embeddings,
        )

        return HuggingFaceEmbeddings(
            model_name=resolved_name,
            model_kwargs={"device": resolved_device, **model_kwargs},
            encode_kwargs=encode_kwargs,
            query_encode_kwargs=encode_kwargs,
        )


@lru_cache(maxsize=1)
def get_embeddings(
    model_name: str | None = None,
    device: str | None = None,
) -> HuggingFaceEmbeddings:
    """Cached factory used by the vector store and the RAG engine.

    The sentence-transformers model is heavy; reuse a single instance per
    process instead of reloading weights on every call.
    """
    return EmbeddingsFactory.create(model_name=model_name, device=device)
