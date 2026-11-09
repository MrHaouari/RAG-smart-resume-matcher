"""
PDF parsing and semantic chunking for resume / job-description documents.

Pipeline step covered here:
  PDF file → raw text (pypdf) → LangChain Documents → overlapping chunks
  via RecursiveCharacterTextSplitter, with rich per-chunk metadata.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from src.config import CHUNK_OVERLAP, CHUNK_SIZE

logger = logging.getLogger(__name__)


class PDFParserError(Exception):
    """Raised when a PDF cannot be read or yields no usable text."""


class PDFParser:
    """Extract text from a PDF and split it into overlapping semantic chunks.

    Designed for CVs and job descriptions: prefers paragraph / line breaks
    before falling back to word-level splits, and attaches provenance
    metadata (source path, page, chunk index) for downstream RAG citations.
    """

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        separators: list[str] | None = None,
    ) -> None:
        """
        Args:
            chunk_size: Target max characters per chunk.
            chunk_overlap: Overlap between consecutive chunks (context continuity).
            separators: Split priority list for RecursiveCharacterTextSplitter.
                Defaults to a CV-friendly hierarchy (paragraph → line → space).
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or [
            "\n\n",  # paragraphs / section breaks
            "\n",  # bullet lines
            ". ",  # sentence boundaries
            " ",  # words
            "",  # characters (last resort)
        ]
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
            is_separator_regex=False,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, pdf_path: str | Path) -> list[Document]:
        """Load a PDF into one LangChain Document per page.

        Args:
            pdf_path: Path to a local PDF file.

        Returns:
            List of Documents whose ``page_content`` is the page text and
            whose ``metadata`` includes source, page number, and page count.

        Raises:
            FileNotFoundError: If the path does not exist.
            PDFParserError: If the file is not a readable PDF or has no text.
        """
        path = self._validate_path(pdf_path)
        reader = self._open_reader(path)
        page_count = len(reader.pages)

        documents: list[Document] = []
        for page_index, page in enumerate(reader.pages):
            text = (page.extract_text() or "").strip()
            if not text:
                logger.warning(
                    "Page %s of %s has no extractable text (scanned image?).",
                    page_index + 1,
                    path.name,
                )
                continue

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": str(path.resolve()),
                        "filename": path.name,
                        "page": page_index + 1,  # 1-indexed for humans
                        "page_index": page_index,  # 0-indexed for machines
                        "total_pages": page_count,
                        "doc_type": "pdf",
                    },
                )
            )

        if not documents:
            raise PDFParserError(
                f"No extractable text found in '{path}'. "
                "The PDF may be empty or image-only (OCR not supported yet)."
            )

        logger.info(
            "Loaded %s page(s) with text from '%s' (%s total pages).",
            len(documents),
            path.name,
            page_count,
        )
        return documents

    def split(self, documents: list[Document]) -> list[Document]:
        """Split page-level Documents into overlapping chunks.

        Preserves original metadata and adds ``chunk_index`` / ``chunk_count``.

        Args:
            documents: Documents typically produced by :meth:`load`.

        Returns:
            Chunked Documents ready for embedding / vector indexing.
        """
        if not documents:
            return []

        raw_chunks = self._splitter.split_documents(documents)
        total = len(raw_chunks)

        chunks: list[Document] = []
        for index, chunk in enumerate(raw_chunks):
            metadata: dict[str, Any] = dict(chunk.metadata)
            metadata.update(
                {
                    "chunk_index": index,
                    "chunk_count": total,
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                    "char_count": len(chunk.page_content),
                }
            )
            chunks.append(
                Document(page_content=chunk.page_content, metadata=metadata)
            )

        logger.info("Split %s page document(s) into %s chunk(s).", len(documents), total)
        return chunks

    def load_and_split(self, pdf_path: str | Path) -> list[Document]:
        """Convenience helper: load a PDF then split it into chunks."""
        return self.split(self.load(pdf_path))

    def extract_text(self, pdf_path: str | Path) -> str:
        """Return the full concatenated text of a PDF (no chunking)."""
        pages = self.load(pdf_path)
        return "\n\n".join(doc.page_content for doc in pages)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_path(pdf_path: str | Path) -> Path:
        path = Path(pdf_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")
        if not path.is_file():
            raise PDFParserError(f"Path is not a file: {path}")
        if path.suffix.lower() != ".pdf":
            raise PDFParserError(f"Expected a .pdf file, got: {path.suffix!r}")
        return path

    @staticmethod
    def _open_reader(path: Path) -> PdfReader:
        try:
            reader = PdfReader(str(path))
        except Exception as exc:  # noqa: BLE001 — surface as domain error
            raise PDFParserError(f"Failed to open PDF '{path}': {exc}") from exc

        if getattr(reader, "is_encrypted", False):
            # Try empty password (common for "restricted" but readable PDFs).
            try:
                unlocked = reader.decrypt("")
            except Exception as exc:  # noqa: BLE001
                raise PDFParserError(
                    f"Encrypted PDF cannot be decrypted: {path}"
                ) from exc
            if not unlocked:
                raise PDFParserError(f"Encrypted PDF requires a password: {path}")

        return reader


# ----------------------------------------------------------------------
# Manual smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )

    if len(sys.argv) < 2:
        print(
            "Usage: python -m src.parsing.pdf_parser <path_to_resume.pdf>\n"
            "Example: python -m src.parsing.pdf_parser data/resumes/sample_cv.pdf"
        )
        sys.exit(1)

    target = Path(sys.argv[1])
    parser = PDFParser()

    try:
        chunks = parser.load_and_split(target)
    except (FileNotFoundError, PDFParserError) as err:
        logger.error("%s", err)
        sys.exit(1)

    print(f"\n=== PDFParser smoke test: {target.name} ===")
    print(f"Chunks produced : {len(chunks)}")
    print(f"Chunk size      : {parser.chunk_size}")
    print(f"Chunk overlap   : {parser.chunk_overlap}")
    print("-" * 60)

    preview_n = min(3, len(chunks))
    for chunk in chunks[:preview_n]:
        meta = chunk.metadata
        preview = chunk.page_content.replace("\n", " ")[:160]
        print(
            f"[chunk {meta['chunk_index'] + 1}/{meta['chunk_count']}] "
            f"page={meta.get('page')} | chars={meta['char_count']}"
        )
        print(f"  {preview}…\n")

    if len(chunks) > preview_n:
        print(f"… {len(chunks) - preview_n} more chunk(s) omitted.")
