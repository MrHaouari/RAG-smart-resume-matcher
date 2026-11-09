"""Local LLM clients (Ollama / ChatOllama)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.llm.ollama_client import (
        build_chat_ollama,
        chat_ollama,
        get_chat_ollama,
        get_structured_matcher,
    )

__all__ = ["build_chat_ollama", "chat_ollama", "get_chat_ollama", "get_structured_matcher"]


def __getattr__(name: str):
    from src.llm import ollama_client as _client

    mapping = {
        "build_chat_ollama": _client.build_chat_ollama,
        "chat_ollama": _client.chat_ollama,
        "get_chat_ollama": _client.get_chat_ollama,
        "get_structured_matcher": _client.get_structured_matcher,
    }
    if name in mapping:
        return mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
