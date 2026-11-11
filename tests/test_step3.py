"""Unit tests for Step 3: MatchReport schema, Ollama client, and RAG Pipeline."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock
from langchain_core.documents import Document
from pydantic import ValidationError

from src.config import settings
from src.llm.ollama_client import build_chat_ollama, get_chat_ollama, chat_ollama
from src.rag.pipeline import DEFAULT_RETRIEVER_K, RAGPipeline, ResumeMatcherRAG
from src.schemas.match_report import MatchReport


class TestStep3(unittest.TestCase):
    """Test suite for schemas, ollama client, and RAG pipeline."""

    def test_match_report_valid(self):
        """Test valid MatchReport instantiation and validation."""
        data = {
            "match_score": 85,
            "matching_skills": ["Python", "LangChain", "ChromaDB"],
            "missing_skills": ["Docker", "Kubernetes"],
            "summary": "Le candidat possède une solide expertise en Python et RAG. Docker reste à acquérir.",
            "recommendations": ["Mettre en avant les projets RAG réalisés en production."],
        }
        report = MatchReport.model_validate(data)
        self.assertEqual(report.match_score, 85)
        self.assertEqual(len(report.matching_skills), 3)
        self.assertEqual(len(report.missing_skills), 2)
        self.assertIn("Python", report.matching_skills)
        self.assertTrue(report.summary.startswith("Le candidat"))

    def test_match_report_invalid_score(self):
        """Test that match_score enforces 0 <= score <= 100."""
        with self.assertRaises(ValidationError):
            MatchReport(
                match_score=105,
                matching_skills=["Python"],
                missing_skills=[],
                summary="Invalid score test",
                recommendations=[],
            )

        with self.assertRaises(ValidationError):
            MatchReport(
                match_score=-5,
                matching_skills=["Python"],
                missing_skills=[],
                summary="Invalid score test",
                recommendations=[],
            )

    def test_match_report_extra_fields_ignored(self):
        """Test that extra fields from LLMs are safely ignored instead of failing validation."""
        data = {
            "match_score": 75,
            "matching_skills": ["Python"],
            "missing_skills": ["Java"],
            "summary": "Bonne adéquation globale.",
            "recommendations": ["Pratiquer Java."],
            "unexpected_extra_field": "some hallucinated metadata",
        }
        report = MatchReport.model_validate(data)
        self.assertEqual(report.match_score, 75)
        self.assertFalse(hasattr(report, "unexpected_extra_field"))

    def test_ollama_client_configuration(self):
        """Test ChatOllama instantiation from settings."""
        client = build_chat_ollama()
        self.assertEqual(client.model, settings.OLLAMA_MODEL)
        self.assertEqual(client.base_url, settings.OLLAMA_BASE_URL)
        self.assertEqual(client.temperature, settings.OLLAMA_TEMPERATURE)

        cached = get_chat_ollama()
        self.assertEqual(cached.model, settings.OLLAMA_MODEL)
        self.assertEqual(chat_ollama.model, settings.OLLAMA_MODEL)

    def test_rag_pipeline_retrieve_and_messages(self):
        """Test RAGPipeline retrieval and prompt composition."""
        mock_vectorstore = MagicMock()
        mock_vectorstore.similarity_search.return_value = [
            Document(
                page_content="Expert Python et LangChain avec 3 ans d'expérience.",
                metadata={"filename": "cv.pdf", "page": 1, "chunk_index": 0},
            ),
            Document(
                page_content="Projets RAG avec ChromaDB et Ollama sur Mac M2.",
                metadata={"filename": "cv.pdf", "page": 1, "chunk_index": 1},
            ),
        ]

        mock_cross_encoder = MagicMock()
        mock_cross_encoder.predict.return_value = [0.95, 0.75]

        mock_llm = MagicMock()
        mock_report = MatchReport(
            match_score=90,
            matching_skills=["Python", "LangChain", "ChromaDB"],
            missing_skills=["Docker"],
            summary="Excellente adéquation avec le profil recherché.",
            recommendations=["Ajouter des détails sur les optimisations Mac M2."],
        )
        mock_llm.invoke.return_value = mock_report

        pipeline = RAGPipeline(
            vector_store=mock_vectorstore,
            structured_llm=mock_llm,
            cross_encoder=mock_cross_encoder,
            retriever_k=DEFAULT_RETRIEVER_K,
        )

        job_desc = "Recherche développeur Python RAG et ChromaDB"
        docs = pipeline.retrieve(job_desc, k=2)
        self.assertEqual(len(docs), 2)
        mock_vectorstore.similarity_search.assert_called_once_with(job_desc, k=pipeline.initial_k)

        report = pipeline.analyze(job_desc)
        self.assertEqual(report.match_score, 90)
        self.assertEqual(report.matching_skills, ["Python", "LangChain", "ChromaDB"])
        mock_llm.invoke.assert_called_once()

    def test_rag_pipeline_two_stage_reranking(self):
        """Test Stage 1 vector search + Stage 2 Cross-Encoder re-ranking."""
        mock_vectorstore = MagicMock()
        doc_a = Document(page_content="Chunk A (weak match)", metadata={"page": 1})
        doc_b = Document(page_content="Chunk B (strong match)", metadata={"page": 2})
        mock_vectorstore.similarity_search.return_value = [doc_a, doc_b]

        mock_cross_encoder = MagicMock()
        mock_cross_encoder.predict.return_value = [0.15, 0.88]

        pipeline = RAGPipeline(
            vector_store=mock_vectorstore,
            structured_llm=MagicMock(),
            cross_encoder=mock_cross_encoder,
            initial_k=15,
            final_k=1,
        )

        job_desc = "Recherche ingénieur IA"
        top_docs = pipeline.retrieve(job_desc)

        mock_vectorstore.similarity_search.assert_called_once_with(job_desc, k=15)
        mock_cross_encoder.predict.assert_called_once()
        self.assertEqual(len(top_docs), 1)
        self.assertEqual(top_docs[0].page_content, "Chunk B (strong match)")
        self.assertAlmostEqual(top_docs[0].metadata["rerank_score"], 0.88)

    def test_rag_pipeline_empty_job_description(self):
        """Test that empty job description raises ValueError."""
        pipeline = RAGPipeline(
            vector_store=MagicMock(),
            structured_llm=MagicMock(),
            cross_encoder=MagicMock(),
        )
        with self.assertRaises(ValueError):
            pipeline.retrieve("   ")

    def test_sanitize_text(self):
        """Test sanitization of control characters, bullets, and whitespace."""
        from src.rag.ingestion import sanitize_text

        dirty_text = (
            "John Doe\x7f - Dev\n\n\n"
            "• Python\n"
            "● LangChain\n"
            "\uf0b7 ChromaDB   with   spaces\n\n\n\n"
            "End."
        )
        cleaned = sanitize_text(dirty_text)
        self.assertNotIn("\x7f", cleaned)
        self.assertNotIn("•", cleaned)
        self.assertNotIn("●", cleaned)
        self.assertIn("- Python", cleaned)
        self.assertIn("- LangChain", cleaned)
        self.assertIn("- ChromaDB with spaces", cleaned)
        self.assertNotIn("\n\n\n", cleaned)

    def test_parent_child_ingestion_chunking(self):
        """Test Parent-Child chunking and parent_id linkage."""
        from src.rag.ingestion import ParentChildIngestion

        ingester = ParentChildIngestion(
            parent_chunk_size=300,
            parent_chunk_overlap=50,
            child_chunk_size=80,
            child_chunk_overlap=15,
        )
        sample_page = Document(
            page_content="Senior Software Engineer. Experience with Python, LangChain, ChromaDB and local LLMs. "
                         "Developed end-to-end RAG applications with high performance on Apple Silicon.",
            metadata={"filename": "cv.pdf", "page": 1},
        )
        result = ingester.create_parent_child_chunks([sample_page])

        self.assertGreater(len(result.parent_documents), 0)
        self.assertGreater(len(result.child_documents), 0)

        # Vérification de l'association parent_id
        for child in result.child_documents:
            pid = child.metadata.get("parent_id")
            self.assertIsNotNone(pid)
            self.assertIn(pid, result.parent_documents)
            # Vérifier que le contenu de l'enfant est bien inclus dans le parent
            parent = result.parent_documents[pid]
            self.assertIn(child.page_content, parent.page_content)

    def test_parent_resolution_and_deduplication(self):
        """Test that Child Chunks resolve to their Parent Document without duplication."""
        parent_id = "parent_test_123"
        parent_doc = Document(
            page_content="Full parent paragraph with complete sentence structure and context.",
            metadata={"parent_id": parent_id, "page": 1, "filename": "cv.pdf"},
        )
        child1 = Document(
            page_content="child snippet 1",
            metadata={"parent_id": parent_id, "rerank_score": 0.95},
        )
        child2 = Document(
            page_content="child snippet 2",
            metadata={"parent_id": parent_id, "rerank_score": 0.75},
        )

        pipeline = RAGPipeline(
            vector_store=MagicMock(),
            structured_llm=MagicMock(),
            cross_encoder=MagicMock(),
            parent_store={parent_id: parent_doc},
        )

        resolved = pipeline.resolve_parents([child1, child2])
        # Two children from the same parent must resolve to exactly ONE parent document
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0].page_content, parent_doc.page_content)
        self.assertAlmostEqual(resolved[0].metadata["rerank_score"], 0.95)

    def test_alias_resumematcher(self):
        """Test that ResumeMatcherRAG is an alias of RAGPipeline."""
        self.assertIs(ResumeMatcherRAG, RAGPipeline)


if __name__ == "__main__":
    unittest.main()
