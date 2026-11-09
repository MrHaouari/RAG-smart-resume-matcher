"""Document parsing utilities (PDF → text → chunks)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.parsing.pdf_parser import PDFParser

__all__ = ["PDFParser"]


def __getattr__(name: str):
    if name == "PDFParser":
        from src.parsing.pdf_parser import PDFParser as _PDFParser

        return _PDFParser
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
