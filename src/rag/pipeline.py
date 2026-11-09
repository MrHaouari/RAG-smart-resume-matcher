"""RAG orchestrator: retrieve resume chunks, then generate a MatchReport.

Pipeline:
  Job Description (text)
    → similarity search on the indexed CV via VectorStoreManager (k=4)
    → strict French prompt against hallucinations
    → ChatOllama.with_structured_output(MatchReport)
    → validated MatchReport Pydantic object
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import JOBS_DIR, RESUMES_DIR, settings
from src.llm.ollama_client import get_structured_matcher
from src.parsing.pdf_parser import PDFParser, PDFParserError
from src.schemas.match_report import MatchReport
from src.vectorstore.chromadb_manager import VectorStoreManager

logger = logging.getLogger(__name__)

DEFAULT_RETRIEVER_K: int = 4

SYSTEM_PROMPT = """Tu es un expert RH et un évaluateur technique senior spécialisé dans l'analyse de candidatures.
Ton objectif est de réaliser une analyse comparative rigoureuse, objective et sans concession entre une fiche de poste et les extraits d'un CV de candidat obtenus par recherche documentaire (RAG).

RÈGLES STRICTES D'ANALYSE — TOLÉRANCE ZÉRO POUR LES HALLUCINATIONS :
1. BASE DE CONNAISSANCE UNIQUE : Fonde ton analyse EXCLUSIVEMENT sur les extraits du CV fournis dans le message utilisateur. N'extrapole JAMAIS.
2. PAS D'INVENTION : N'invente aucune compétence, technologie, expérience, entreprise, diplôme, date ou métrique qui n'apparaît pas mot pour mot ou de manière irréfutable dans les extraits.
3. matching_skills :
   - Inclus UNIQUEMENT les compétences, outils ou savoir-faire qui figurent EXPLICITEMENT dans les extraits du CV ET qui sont exigés ou pertinents au vu de la fiche de poste.
   - Ne présuppose pas qu'un candidat maîtrise un outil simplement parce qu'il maîtrise un outil similaire.
4. missing_skills :
   - Liste les exigences, compétences ou technologies indispensables demandées dans la fiche de poste qui sont ABSENTES ou introuvables dans les extraits du CV.
   - Ne liste pas une compétence manquante si elle n'était pas mentionnée ou requise par la fiche de poste.
5. match_score :
   - Évalue de manière honnête et conservatrice l'adéquation sur une échelle de 0 à 100.
   - Si les compétences clés sont absentes, le score doit être sévère (< 50).
   - En cas d'ambiguïté ou d'information parcellaire, sois conservateur.
6. summary :
   - Rédige une synthèse factuelle de 3 à 6 phrases en français, résumant le profil au regard du poste et justifiant le score sans détour ni complaisance.
7. recommendations :
   - Propose des conseils constructifs et exploitables pour adapter le CV à ce poste (reformulation, mise en valeur de projets justifiés, points à clarifier lors d'un entretien).
   - N'incite JAMAIS le candidat à falsifier son parcours.
8. FORMAT OBLIGATOIRE :
   - Ta réponse doit strictement respecter le schéma structuré MatchReport.
"""


class RAGPipeline:
    """Orchestrateur RAG pour l'évaluation CV vs Fiche de poste."""

    def __init__(
        self,
        vector_store: VectorStoreManager | None = None,
        structured_llm=None,
        retriever_k: int = DEFAULT_RETRIEVER_K,
    ) -> None:
        """
        Args:
            vector_store: Gestionnaire ChromaDB pour les chunks de CV.
            structured_llm: Client LLM configuré avec .with_structured_output(MatchReport).
            retriever_k: Nombre de chunks les plus pertinents à extraire (défaut: 4).
        """
        if retriever_k <= 0:
            raise ValueError("retriever_k doit être un entier strictement positif (> 0).")
        self.vector_store = vector_store or VectorStoreManager()
        self.structured_llm = structured_llm or get_structured_matcher()
        self.retriever_k = retriever_k

    def index_resume(self, pdf_path: str | Path, reset: bool = True) -> int:
        """Extrait les chunks d'un CV PDF et les indexe dans ChromaDB.

        Args:
            pdf_path: Chemin vers le fichier PDF du CV.
            reset: Si True, réinitialise la collection pour isoler l'analyse du candidat.

        Returns:
            Nombre de chunks indexés.
        """
        parser = PDFParser()
        chunks = parser.load_and_split(pdf_path)
        if reset:
            self.vector_store.reset_collection()
        ids = self.vector_store.add_documents(chunks)
        logger.info("CV '%s' indexé avec succès (%d chunk(s)).", pdf_path, len(ids))
        return len(ids)

    def retrieve(self, job_description: str, k: int | None = None) -> list[Document]:
        """Interroge VectorStoreManager pour récupérer les k chunks les plus pertinents.

        Args:
            job_description: Texte complet de la fiche de poste.
            k: Nombre de chunks à récupérer (par défaut self.retriever_k = 4).

        Returns:
            Liste des documents pertinents avec leurs métadonnées.
        """
        query = job_description.strip()
        if not query:
            raise ValueError("La fiche de poste (job_description) ne peut pas être vide.")
        top_k = k if k is not None else self.retriever_k
        return self.vector_store.similarity_search(query, k=top_k)

    def analyze(
        self,
        job_description: str,
        k: int | None = None,
    ) -> MatchReport:
        """Exécute la recherche vectorielle puis génère le MatchReport structuré.

        Args:
            job_description: Texte complet de la fiche de poste.
            k: Optionnel pour surcharger le nombre de chunks récupérés.

        Returns:
            Une instance validée de ``MatchReport``.
        """
        docs = self.retrieve(job_description, k=k)
        messages = self._build_messages(job_description, docs)
        logger.info(
            "Appel du LLM structuré avec %d chunk(s) de contexte RAG.",
            len(docs),
        )
        raw_output = self.structured_llm.invoke(messages)

        if isinstance(raw_output, MatchReport):
            return raw_output

        # Sécurité supplémentaire au cas où le retour est un dict ou objet compatible
        if isinstance(raw_output, dict):
            return MatchReport.model_validate(raw_output)

        return MatchReport.model_validate(raw_output)

    def analyze_resume(
        self,
        pdf_path: str | Path,
        job_description: str,
        k: int | None = None,
    ) -> MatchReport:
        """Méthode de bout en bout : indexation du CV PDF puis analyse RAG."""
        self.index_resume(pdf_path, reset=True)
        return self.analyze(job_description, k=k)

    @staticmethod
    def _format_context(documents: list[Document]) -> str:
        """Formate les chunks de CV en contexte propre et balisé pour le LLM."""
        if not documents:
            return (
                "[Aucun extrait de CV n'a été trouvé dans la base vectorielle. "
                "Le score doit être 0 ou très faible, matching_skills vide, "
                "et toutes les compétences de la fiche dans missing_skills.]"
            )

        blocks: list[str] = []
        for index, doc in enumerate(documents, start=1):
            page = doc.metadata.get("page", "?")
            chunk_index = doc.metadata.get("chunk_index", "?")
            filename = doc.metadata.get("filename", "cv.pdf")
            header = f"--- Extrait #{index} | Source: {filename} (page {page}, chunk {chunk_index}) ---"
            blocks.append(f"{header}\n{doc.page_content.strip()}")
        return "\n\n".join(blocks)

    def _build_messages(
        self,
        job_description: str,
        documents: list[Document],
    ) -> list[SystemMessage | HumanMessage]:
        """Construit la liste des messages (System + Human) avec balisage strict."""
        formatted_context = self._format_context(documents)
        human_content = (
            "FICHE DE POSTE CIBLE (Job Description) :\n"
            "=========================================\n"
            f"{job_description.strip()}\n"
            "=========================================\n\n"
            "EXTRAITS PERTINENTS DU CV DU CANDIDAT (Contexte RAG - Seule source autorisée) :\n"
            "=========================================\n"
            f"{formatted_context}\n"
            "=========================================\n\n"
            "Génère le rapport d'analyse d'adéquation strictement conforme au schéma MatchReport."
        )
        return [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=human_content),
        ]


# Alias alternatif
ResumeMatcherRAG = RAGPipeline


# ----------------------------------------------------------------------
# Smoke test complet : Parsing CV -> Indexation -> RAG Analysis -> MatchReport
# ----------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    default_pdf = RESUMES_DIR / "sample_cv.pdf"
    pdf_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_pdf

    sample_job_path = JOBS_DIR / "sample_job.txt"
    if sample_job_path.exists():
        job_description = sample_job_path.read_text(encoding="utf-8")
    else:
        job_description = """
Développeur Python / RAG — CDI — Paris (hybride)
Nous recherchons un(e) Software Engineer pour concevoir des applications RAG locales.
Compétences exigées :
- Python avancé
- LangChain & ChromaDB
- LLMs locaux (Ollama)
- Embeddings sentence-transformers
- Pydantic v2
""".strip()

    print("\n" + "=" * 65)
    print("🚀 SMOKE TEST : Smart Resume Matcher & Career Assistant (Étape 3)")
    print("=" * 65)
    print(f"📄 CV Path     : {pdf_path}")
    print(f"🤖 LLM Model   : {settings.OLLAMA_MODEL} @ {settings.OLLAMA_BASE_URL}")
    print(f"🧠 Embeddings  : {settings.EMBEDDING_MODEL_NAME}")
    print(f"🎯 Retriever k : {DEFAULT_RETRIEVER_K}")
    print("-" * 65)

    if not pdf_path.exists():
        logger.error("Le fichier CV spécifié n'existe pas : %s", pdf_path)
        sys.exit(1)

    try:
        pipeline = RAGPipeline(retriever_k=DEFAULT_RETRIEVER_K)

        print("\n[1/3] Indexation du CV dans ChromaDB...")
        num_chunks = pipeline.index_resume(pdf_path, reset=True)
        print(f"      ✅ {num_chunks} chunks indexés avec succès.")

        print("\n[2/3] Recherche vectorielle (k=4) et inférence LLM...")
        report: MatchReport = pipeline.analyze(job_description)
        print("      ✅ Rapport structuré obtenu avec succès.")

        print("\n[3/3] Résultat du MatchReport :")
        print("=" * 65)
        print(f"📊 Score de correspondance : {report.match_score}/100")
        print(f"\n✅ Compétences correspondantes ({len(report.matching_skills)}) :")
        for skill in report.matching_skills:
            print(f"   • {skill}")

        print(f"\n❌ Compétences manquantes ({len(report.missing_skills)}) :")
        for skill in report.missing_skills:
            print(f"   • {skill}")

        print(f"\n💡 Recommandations ({len(report.recommendations)}) :")
        for rec in report.recommendations:
            print(f"   • {rec}")

        print(f"\n📝 Synthèse factuelle :\n   {report.summary}")
        print("=" * 65)

        print("\n📦 Export JSON validé Pydantic v2 :")
        print(report.model_dump_json(indent=2))
        print("\n🎉 Smoke test Étape 3 validé avec succès !")

    except (FileNotFoundError, PDFParserError) as err:
        logger.error("Erreur de parsing : %s", err)
        sys.exit(1)
    except Exception as err:
        logger.error("Erreur durant le pipeline RAG : %s", err, exc_info=True)
        sys.exit(1)
