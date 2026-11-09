"""Local Ollama chat client (no cloud API).

Talks to the daemon at ``OLLAMA_BASE_URL`` (default http://localhost:11434)
using ``OLLAMA_MODEL`` (llama3.2) with ``OLLAMA_TEMPERATURE`` (0.1)
for deterministic and structured JSON outputs.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from langchain_ollama import ChatOllama

from src.config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TEMPERATURE, settings
from src.schemas.match_report import MatchReport

logger = logging.getLogger(__name__)


def build_chat_ollama(
    model: str | None = None,
    base_url: str | None = None,
    temperature: float | None = None,
) -> ChatOllama:
    """Instantiate a local ``ChatOllama`` from project settings.

    Args:
        model: Ollama model tag. Defaults to ``OLLAMA_MODEL`` (e.g. 'llama3.2').
        base_url: Ollama daemon URL. Defaults to ``OLLAMA_BASE_URL`` ('http://localhost:11434').
        temperature: Sampling temperature. Defaults to ``OLLAMA_TEMPERATURE`` (0.1).

    Returns:
        Configured ``ChatOllama`` client instance.
    """
    resolved_model = model or OLLAMA_MODEL
    resolved_url = base_url or OLLAMA_BASE_URL
    resolved_temp = (
        OLLAMA_TEMPERATURE if temperature is None else temperature
    )

    logger.info(
        "Initializing ChatOllama client [model=%s, url=%s, temperature=%.2f]",
        resolved_model,
        resolved_url,
        resolved_temp,
    )

    return ChatOllama(
        model=resolved_model,
        base_url=resolved_url,
        temperature=resolved_temp,
    )


@lru_cache(maxsize=1)
def get_chat_ollama() -> ChatOllama:
    """Return a process-wide cached ``ChatOllama`` client instance."""
    return build_chat_ollama()


# Ready-to-use ChatOllama instance configured from src/config.py
chat_ollama: ChatOllama = build_chat_ollama()


def get_structured_matcher(llm: ChatOllama | None = None):
    """ChatOllama client bound to the ``MatchReport`` Pydantic schema.

    Forces Ollama to output structured, schema-compliant JSON validated
    as a ``MatchReport`` Pydantic instance.
    """
    target_llm = llm or get_chat_ollama()
    return target_llm.with_structured_output(MatchReport)
