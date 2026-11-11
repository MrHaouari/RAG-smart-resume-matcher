"""Module d'ingestion avancée avec Sanitization et stratégie Parent-Child Chunking.

Stratégie Parent-Document Retriever :
1. Extraction & Sanitization :
   - Nettoyage rigoureux du texte brut PDF (caractères de contrôle non imprimables,
     puces corrompues, césures de fin de ligne, normalisation des espaces/retours).
2. Découpage Parent-Child :
   - Parent Chunks (~800-1000 caractères) : sections complètes et cohérentes,
     garantissant l'absence de phrases coupées dans le contexte final du LLM.
   - Child Chunks (~150-200 caractères) : micro-segments denses, découpés
     strictement aux limites de mots, optimisés pour la recherche vectorielle (ChromaDB)
     et le scoring de haute précision par le Re-ranker (Cross-Encoder).
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from src.config import (
    CHILD_CHUNK_OVERLAP,
    CHILD_CHUNK_SIZE,
    PARENT_CHUNK_OVERLAP,
    PARENT_CHUNK_SIZE,
)
from src.parsing.pdf_parser import PDFParserError

logger = logging.getLogger(__name__)


def sanitize_text(text: str) -> str:
    """Nettoie et normalise rigoureusement le texte brut extrait d'un document.

    Opérations effectuées :
    - Élimine les caractères de contrôle non imprimables (ex: \\x7f, \\x00-\\x1f sauf \\n et \\t).
    - Remplace les espaces insécables et marques d'ordre d'octets (BOM).
    - Convertit les puces et symboles spéciaux (•, ●, ▪, \\uf0b7, etc.) en tirets propres '- '.
    - Supprime les césures de fin de ligne (mots coupés par des tirets : 'déve-\\nloppeur' -> 'développeur').
    - Normalise les espaces horizontaux consécutifs en un espace unique.
    - Nettoie chaque ligne et limite les sauts de ligne consécutifs à 2 au maximum.

    Args:
        text: Chaîne de caractères brute extraite du PDF.

    Returns:
        Chaîne nettoyée, normalisée et prête pour la tokenisation.
    """
    if not text:
        return ""

    # 1. Suppression du BOM et normalisation des espaces insécables
    cleaned = text.replace("\ufeff", "").replace("\xa0", " ")

    # 2. Suppression des caractères de contrôle non imprimables (garde \n et \t)
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", cleaned)

    # 3. Reconnexion des mots coupés par un saut de ligne avec tiret (césure)
    cleaned = re.sub(r"(\b\w+)-\n(\w+\b)", r"\1\2", cleaned)

    # 4. Normalisation des puces diverses en tirets standards
    bullet_pattern = r"[\u2022\u2023\u25e6\u2043\u2219\u25cf\u25cb\u25aa\u25ab\uf0b7•●▪▫◦◆]"
    cleaned = re.sub(bullet_pattern, "\n- ", cleaned)

    # 5. Remplacement des tabulations par un espace
    cleaned = cleaned.replace("\t", " ")

    # 6. Remplacement des espaces consécutifs par un seul espace
    cleaned = re.sub(r"[ ]{2,}", " ", cleaned)

    # 7. Élimination des espaces superflus en début/fin de chaque ligne
    lines = [line.strip() for line in cleaned.splitlines()]
    cleaned = "\n".join(lines)

    # 8. Normalisation des doubles tirets ou tirets mal espacés
    cleaned = re.sub(r"^-\s*-\s*", "- ", cleaned, flags=re.MULTILINE)

    # 9. Réduction des sauts de ligne multiples consécutifs (max 2)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()


@dataclass(frozen=True)
class IngestionResult:
    """Résultat de l'ingestion structurée Parent-Child.

    Attributes:
        parent_documents: Dictionnaire associant chaque `parent_id` à son `Document` parent complet.
        child_documents: Liste de micro-chunks `Document` destinés à être vectorisés dans ChromaDB.
    """

    parent_documents: dict[str, Document]
    child_documents: list[Document]


class ParentChildIngestion:
    """Gère l'extraction PDF, le nettoyage et le partitionnement hiérarchique Parent-Child."""

    def __init__(
        self,
        parent_chunk_size: int = PARENT_CHUNK_SIZE,
        parent_chunk_overlap: int = PARENT_CHUNK_OVERLAP,
        child_chunk_size: int = CHILD_CHUNK_SIZE,
        child_chunk_overlap: int = CHILD_CHUNK_OVERLAP,
    ) -> None:
        """
        Args:
            parent_chunk_size: Taille des sections parent (~800-1000 caractères).
            parent_chunk_overlap: Recouvrement entre parents pour préserver la continuité.
            child_chunk_size: Taille des micro-chunks enfants (~150-200 caractères).
            child_chunk_overlap: Recouvrement entre enfants.
        """
        if parent_chunk_size <= child_chunk_size:
            raise ValueError(
                f"parent_chunk_size ({parent_chunk_size}) doit être supérieur à child_chunk_size ({child_chunk_size})."
            )

        self.parent_chunk_size = parent_chunk_size
        self.parent_chunk_overlap = parent_chunk_overlap
        self.child_chunk_size = child_chunk_size
        self.child_chunk_overlap = child_chunk_overlap

        # Splitter pour les Parent Chunks : préserve les paragraphes et phrases
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.parent_chunk_size,
            chunk_overlap=self.parent_chunk_overlap,
            separators=["\n\n", "\n", ". ", " "],
        )

        # Splitter pour les Child Chunks : ne coupe JAMAIS de mots (pas de séparateur vide "")
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.child_chunk_size,
            chunk_overlap=self.child_chunk_overlap,
            separators=["\n\n", "\n", " "],
        )

    def extract_and_sanitize_pdf(self, pdf_path: str | Path) -> list[Document]:
        """Extrait le texte page par page d'un PDF et applique la sanitization.

        Args:
            pdf_path: Chemin absolu ou relatif vers le document PDF.

        Returns:
            Liste de documents par page nettoyés.
        """
        path = Path(pdf_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Fichier PDF introuvable : {path}")

        try:
            reader = PdfReader(str(path))
        except Exception as err:
            raise PDFParserError(f"Impossible de lire le PDF '{path.name}': {err}") from err

        if reader.is_encrypted:
            raise PDFParserError(f"Le document PDF '{path.name}' est protégé par mot de passe.")

        documents: list[Document] = []
        for index, page in enumerate(reader.pages, start=1):
            try:
                raw_text = page.extract_text() or ""
            except Exception as err:
                logger.warning("Erreur lors de l'extraction de la page %d du fichier %s : %s", index, path.name, err)
                raw_text = ""

            cleaned_text = sanitize_text(raw_text)
            if cleaned_text:
                documents.append(
                    Document(
                        page_content=cleaned_text,
                        metadata={
                            "source": str(path),
                            "filename": path.name,
                            "page": index,
                        },
                    )
                )

        if not documents:
            raise PDFParserError(
                f"Aucun texte exploitable n'a pu être extrait du PDF '{path.name}' "
                "(document vide ou composé uniquement d'images scannées sans couche OCR)."
            )

        logger.info("PDF '%s' extrait et assaini : %d page(s) exploitable(s).", path.name, len(documents))
        return documents

    def create_parent_child_chunks(
        self,
        page_documents: Sequence[Document],
    ) -> IngestionResult:
        """Découpe les pages en Parent Chunks puis génère les Child Chunks associés.

        Args:
            page_documents: Documents de pages nettoyés.

        Returns:
            IngestionResult avec store des parents et micro-chunks enfants.
        """
        parent_store: dict[str, Document] = {}
        all_children: list[Document] = []

        global_parent_idx = 0
        global_child_idx = 0

        for page_doc in page_documents:
            # 1. Découpage en sections parent
            parent_chunks = self.parent_splitter.split_documents([page_doc])

            for p_chunk in parent_chunks:
                global_parent_idx += 1
                parent_id = f"parent_{uuid.uuid4().hex[:8]}_{global_parent_idx}"

                # Métadonnées du parent
                parent_metadata = dict(p_chunk.metadata)
                parent_metadata.update({
                    "parent_id": parent_id,
                    "parent_index": global_parent_idx,
                    "is_parent": True,
                })
                parent_doc = Document(
                    page_content=p_chunk.page_content,
                    metadata=parent_metadata,
                )
                parent_store[parent_id] = parent_doc

                # 2. Découpage du parent en micro-chunks enfants
                child_texts = self.child_splitter.split_text(p_chunk.page_content)
                for c_text in child_texts:
                    global_child_idx += 1
                    child_metadata = dict(parent_metadata)
                    child_metadata.update({
                        "parent_id": parent_id,
                        "child_index": global_child_idx,
                        "is_child": True,
                        "is_parent": False,
                    })
                    all_children.append(
                        Document(
                            page_content=c_text,
                            metadata=child_metadata,
                        )
                    )

        logger.info(
            "Partitionnement Parent-Child achevé : %d Parent Chunks -> %d Child Chunks générés.",
            len(parent_store),
            len(all_children),
        )
        return IngestionResult(parent_documents=parent_store, child_documents=all_children)

    def ingest_pdf(self, pdf_path: str | Path) -> IngestionResult:
        """Méthode tout-en-un : extraction, sanitization et partitionnement hiérarchique."""
        pages = self.extract_and_sanitize_pdf(pdf_path)
        return self.create_parent_child_chunks(pages)
