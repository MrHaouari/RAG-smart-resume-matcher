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
            retriever_k=DEFAULT_RETRIEVER_K,
        )

        job_desc = "Recherche développeur Python RAG et ChromaDB"
        docs = pipeline.retrieve(job_desc, k=2)
        self.assertEqual(len(docs), 2)
        mock_vectorstore.similarity_search.assert_called_once_with(job_desc, k=2)

        report = pipeline.analyze(job_desc)
        self.assertEqual(report.match_score, 90)
        self.assertEqual(report.matching_skills, ["Python", "LangChain", "ChromaDB"])
        mock_llm.invoke.assert_called_once()

    def test_rag_pipeline_empty_job_description(self):
        """Test that empty job description raises ValueError."""
        pipeline = RAGPipeline(vector_store=MagicMock(), structured_llm=MagicMock())
        with self.assertRaises(ValueError):
            pipeline.retrieve("   ")

    def test_alias_resumematcher(self):
        """Test that ResumeMatcherRAG is an alias of RAGPipeline."""
        self.assertIs(ResumeMatcherRAG, RAGPipeline)


if __name__ == "__main__":
    unittest.main()
