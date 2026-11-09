"""Persistent local ChromaDB manager for resume chunks.

Wraps ``langchain_chroma.Chroma`` so the rest of the app never talks to the
vector database directly. Persistence lives on disk under
``settings.chroma_persist_path`` (``./chroma_db`` by default).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Sequence
from uuid import uuid4

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma

from src.config import settings
from src.embeddings.embeddings_factory import get_embeddings

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Index resume chunks and run similarity search against a job description."""

    def __init__(
        self,
        persist_directory: str | Path | None = None,
        collection_name: str | None = None,
        embeddings: Embeddings | None = None,
    ) -> None:
        """
        Args:
            persist_directory: On-disk Chroma path. Defaults to
                ``settings.chroma_persist_path``.
            collection_name: Chroma collection. Defaults to
                ``settings.CHROMA_COLLECTION_NAME``.
            embeddings: Embedding model. Defaults to the cached local
                HuggingFace encoder.
        """
        self.persist_directory = str(
            Path(persist_directory).resolve()
            if persist_directory is not None
            else settings.chroma_persist_path
        )
        self.collection_name = collection_name or settings.CHROMA_COLLECTION_NAME
        self.embeddings = embeddings or get_embeddings()
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        self._store = self._build_store()
        logger.info(
            "Chroma collection '%s' ready at %s",
            self.collection_name,
            self.persist_directory,
        )

    @property
    def store(self) -> Chroma:
        """Underlying LangChain Chroma vector store."""
        return self._store

    def add_documents(self, documents: Sequence[Document]) -> list[str]:
        """Embed and persist resume chunks.

        Args:
            documents: Chunks produced by ``PDFParser.load_and_split``.

        Returns:
            The Chroma ids assigned to the inserted documents.
        """
        if not documents:
            logger.warning("add_documents called with an empty list; nothing to index.")
            return []

        ids = [_document_id(doc, index) for index, doc in enumerate(documents)]
        inserted = self._store.add_documents(documents=list(documents), ids=ids)
        logger.info(
            "Indexed %s document(s) into collection '%s'.",
            len(documents),
            self.collection_name,
        )
        return list(inserted)

    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        """Return the ``k`` resume chunks closest to ``query``.

        Args:
            query: Typically the job description (or a focused requirement).
            k: Number of chunks to retrieve.

        Returns:
            Ranked ``Document`` list with original parser metadata.
        """
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")
        if k <= 0:
            raise ValueError("k must be > 0")

        results = self._store.similarity_search(query.strip(), k=k)
        logger.info("similarity_search(k=%s) returned %s hit(s).", k, len(results))
        return results

    def similarity_search_with_score(
        self, query: str, k: int = 4
    ) -> list[tuple[Document, float]]:
        """Same as :meth:`similarity_search` but includes distance scores."""
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")
        if k <= 0:
            raise ValueError("k must be > 0")
        return self._store.similarity_search_with_score(query.strip(), k=k)

    def reset_collection(self) -> None:
        """Drop all vectors so a new CV analysis starts from a clean index."""
        self._store.reset_collection()
        logger.info("Reset Chroma collection '%s'.", self.collection_name)

    def _build_store(self) -> Chroma:
        return Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
            # Cosine space matches L2-normalized MiniLM embeddings.
            collection_metadata={"hnsw:space": "cosine"},
        )


def _document_id(document: Document, fallback_index: int) -> str:
    """Stable-enough id from parser metadata, with a UUID suffix for uniqueness."""
    filename = document.metadata.get("filename", "document")
    page = document.metadata.get("page", 0)
    chunk_index = document.metadata.get("chunk_index", fallback_index)
    return f"{filename}::p{page}::c{chunk_index}::{uuid4().hex[:8]}"


# ----------------------------------------------------------------------
# Manual smoke test: PDFParser → embeddings → Chroma
# ----------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )

    from src.parsing.pdf_parser import PDFParser, PDFParserError

    default_pdf = Path(__file__).resolve().parents[2] / "data" / "resumes" / "sample_cv.pdf"
    pdf_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_pdf

    print(f"\n=== VectorStoreManager smoke test ===")
    print(f"PDF              : {pdf_path}")
    print(f"Embeddings       : {settings.EMBEDDING_MODEL_NAME}")
    print(f"Collection       : {settings.CHROMA_COLLECTION_NAME}")
    print(f"Persist dir      : {settings.chroma_persist_path}")
    print("-" * 60)

    parser = PDFParser()
    try:
        chunks = parser.load_and_split(pdf_path)
    except (FileNotFoundError, PDFParserError) as err:
        logger.error("%s", err)
        sys.exit(1)

    manager = VectorStoreManager()
    manager.reset_collection()
    ids = manager.add_documents(chunks)
    print(f"Indexed chunks   : {len(ids)}")

    query = (
        "We are hiring a Software Engineer with Python, RAG, LangChain "
        "and ChromaDB experience on Apple Silicon."
    )
    hits = manager.similarity_search(query, k=min(4, len(chunks)))

    print(f"Query            : {query}")
    print(f"Hits             : {len(hits)}")
    print("-" * 60)
    for rank, doc in enumerate(hits, start=1):
        preview = doc.page_content.replace("\n", " ")[:180]
        print(
            f"[{rank}] page={doc.metadata.get('page')} "
            f"chunk={doc.metadata.get('chunk_index')} "
            f"chars={doc.metadata.get('char_count')}"
        )
        print(f"    {preview}\n")
