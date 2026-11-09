"""Pydantic schemas for structured RAG outputs."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.schemas.match_report import MatchReport

__all__ = ["MatchReport"]


def __getattr__(name: str):
    if name == "MatchReport":
        from src.schemas.match_report import MatchReport as _MatchReport

        return _MatchReport
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
