"""Advanced RAG orchestrator with Parent-Document Retriever and Cross-Encoder Re-ranking.

Architecture :
  Job Description (text)
    │
    ▼ [Étape 1: Fast Retrieval sur Micro-Chunks Enfants]
  ChromaDB Vector Store (Bi-Encoder embeddings)
    → Recherche vectorielle sur les Child Chunks denses (k_initial = 15)
    │
    ▼ [Étape 2: Precision Re-ranking via Cross-Encoder]
  Cross-Encoder (cross-encoder/ms-marco-MiniLM-L-6-v2)
    → Évaluation de l'attention croisée complète (job_description, child_chunk)
    → Tri et sélection des Top-4 Child Chunks les plus pertinents
    │
    ▼ [Étape 3: Résolution Parent-Document (Deduplication)]
  Parent Store (Mémoire / Ingestion)
    → Récupération des Parent Chunks complets (~800-1000 caractères)
    → Élimination des doublons de parents tout en préservant le meilleur score
    │
    ▼ [Étape 4: Synthèse Factuelle sans Hallucination]
  Structured LLM (ChatOllama llama3.2 with MatchReport schema)
    → Injection des sections parentes complètes et non tronquées
    → Production d'un MatchReport Pydantic v2 garanti
"""

from __future__ import annotations

import logging
import sys
from functools import lru_cache
from pathlib import Path
from typing import Sequence

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from sentence_transformers import CrossEncoder

from src.config import (
    CROSS_ENCODER_MODEL_NAME,
    FINAL_RERANK_K,
    INITIAL_RETRIEVAL_K,
    JOBS_DIR,
    RESUMES_DIR,
    settings,
)
from src.embeddings.embeddings_factory import resolve_torch_device
from src.llm.ollama_client import get_structured_matcher
from src.parsing.pdf_parser import PDFParserError
from src.rag.ingestion import ParentChildIngestion
from src.schemas.match_report import MatchReport
from src.vectorstore.chromadb_manager import VectorStoreManager

logger = logging.getLogger(__name__)

# Valeurs par défaut pour le pipeline
DEFAULT_INITIAL_K: int = getattr(settings, "INITIAL_RETRIEVAL_K", 15)
DEFAULT_FINAL_K: int = getattr(settings, "FINAL_RERANK_K", 4)
DEFAULT_RETRIEVER_K: int = DEFAULT_FINAL_K  # Rétrocompatibilité

SYSTEM_PROMPT = """Tu es un expert RH et un évaluateur technique senior spécialisé dans l'analyse de candidatures.
Ton objectif est de réaliser une analyse comparative rigoureuse, objective et sans concession entre une fiche de poste et les extraits d'un CV de candidat obtenus par recherche documentaire avancée (Parent-Document Retriever + Re-ranking Cross-Encoder).

RÈGLES STRICTES D'ANALYSE — TOLÉRANCE ZÉRO POUR LES HALLUCINATIONS :
1. BASE DE CONNAISSANCE UNIQUE : Fonde ton analyse EXCLUSIVEMENT sur les extraits de sections complètes de CV fournis dans le message utilisateur. N'extrapole JAMAIS.
2. PAS D'INVENTION : N'invente aucune compétence, technologie, expérience, entreprise, diplôme, date ou métrique qui n'apparaît pas mot pour mot ou de manière irréfutable dans les extraits.
3. CONTEXTE PARENT COMPLET : Les extraits fournis sont des sections parentes complètes associées aux passages jugés les plus pertinents par le Re-ranker. Analyse l'intégralité du paragraphe sans supposer d'informations manquantes.
4. matching_skills :
   - Inclus UNIQUEMENT les compétences, outils ou savoir-faire qui figurent EXPLICITEMENT dans les extraits du CV ET qui sont exigés ou pertinents au vu de la fiche de poste.
   - Ne présuppose pas qu'un candidat maîtrise un outil simplement parce qu'il maîtrise un outil similaire.
5. missing_skills :
   - Liste les exigences, compétences ou technologies indispensables demandées dans la fiche de poste qui sont ABSENTES ou introuvables dans les extraits du CV.
   - Ne liste pas une compétence manquante si elle n'était pas mentionnée ou requise par la fiche de poste.
6. match_score :
   - Évalue de manière honnête et conservatrice l'adéquation sur une échelle de 0 à 100.
   - Si les compétences clés sont absentes, le score doit être sévère (< 50).
   - En cas d'ambiguïté ou d'information parcellaire, sois conservateur.
7. summary :
   - Rédige une synthèse factuelle de 3 à 6 phrases en français, résumant le profil au regard du poste et justifiant le score sans détour ni complaisance.
8. recommendations :
   - Propose des conseils constructifs et exploitables pour adapter le CV à ce poste (reformulation, mise en valeur de projets justifiés, points à clarifier lors d'un entretien).
   - N'incite JAMAIS le candidat à falsifier son parcours.
9. FORMAT OBLIGATOIRE :
   - Ta réponse doit strictement respecter le schéma structuré MatchReport.
"""


@lru_cache(maxsize=1)
def get_cross_encoder(
    model_name: str | None = None,
    device: str | None = None,
) -> CrossEncoder:
    """Instancie et met en cache le modèle Cross-Encoder pour le re-ranking.

    Args:
        model_name: Identifiant HuggingFace du Cross-Encoder.
        device: 'mps', 'cuda' ou 'cpu' (détecté automatiquement si None).

    Returns:
        Instance de CrossEncoder prête pour l'inférence locale.
    """
    resolved_model = model_name or getattr(
        settings, "CROSS_ENCODER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    resolved_device = device or resolve_torch_device()
    logger.info(
        "Chargement du Cross-Encoder '%s' sur le device '%s'...",
        resolved_model,
        resolved_device,
    )
    return CrossEncoder(model_name=resolved_model, device=resolved_device)


class RAGPipeline:
    """Orchestrateur Advanced RAG combinant Parent-Document Retriever et Re-ranking."""

    def __init__(
        self,
        vector_store: VectorStoreManager | None = None,
        structured_llm=None,
        cross_encoder: CrossEncoder | None = None,
        ingester: ParentChildIngestion | None = None,
        initial_k: int = DEFAULT_INITIAL_K,
        final_k: int = DEFAULT_FINAL_K,
        retriever_k: int | None = None,
        parent_store: dict[str, Document] | None = None,
    ) -> None:
        """
        Args:
            vector_store: Gestionnaire ChromaDB pour les micro-chunks enfants.
            structured_llm: Client LLM configuré avec .with_structured_output(MatchReport).
            cross_encoder: Modèle Cross-Encoder pour le re-ranking.
            ingester: Gestionnaire d'ingestion et sanitization Parent-Child.
            initial_k: Nombre de candidats enfants extraits de ChromaDB (défaut: 15).
            final_k: Nombre de chunks retenus après re-ranking (défaut: 4).
            retriever_k: Paramètre rétrocompatible pour final_k.
            parent_store: Dictionnaire optionnel contenant les documents parents complets.
        """
        if retriever_k is not None:
            final_k = retriever_k
            initial_k = max(initial_k, final_k)

        if initial_k <= 0 or final_k <= 0:
            raise ValueError("initial_k et final_k doivent être des entiers strictement positifs (> 0).")
        if final_k > initial_k:
            initial_k = final_k

        self.vector_store = vector_store or VectorStoreManager()
        self.structured_llm = structured_llm or get_structured_matcher()
        self.cross_encoder = cross_encoder or get_cross_encoder()
        self.ingester = ingester or ParentChildIngestion()
        self.initial_k = initial_k
        self.final_k = final_k
        self.retriever_k = final_k
        self.parent_store: dict[str, Document] = parent_store or {}

    def index_resume(self, pdf_path: str | Path, reset: bool = True) -> int:
        """Nettoie le CV, le partitionne en Parent/Child et indexe les enfants dans ChromaDB.

        Args:
            pdf_path: Chemin vers le fichier PDF du CV.
            reset: Si True, réinitialise la collection ChromaDB et le store de parents.

        Returns:
            Nombre de micro-chunks enfants indexés.
        """
        ingest_result = self.ingester.ingest_pdf(pdf_path)

        if reset:
            self.vector_store.reset_collection()
            self.parent_store.clear()

        # Enregistrement des documents parents complets
        self.parent_store.update(ingest_result.parent_documents)

        # Indexation vectorielle des micro-chunks enfants dans ChromaDB
        ids = self.vector_store.add_documents(ingest_result.child_documents)
        logger.info(
            "Ingestion achevée pour '%s' : %d parents conservés, %d micro-chunks indexés.",
            pdf_path,
            len(ingest_result.parent_documents),
            len(ids),
        )
        return len(ids)

    def rerank_children(
        self,
        query: str,
        child_documents: Sequence[Document],
        top_k: int | None = None,
    ) -> list[Document]:
        """Évalue la pertinence croisée (Cross-Encoder) et trie les Child Chunks.

        Args:
            query: Texte de la fiche de poste.
            child_documents: Micro-chunks candidats issus de ChromaDB.
            top_k: Nombre maximal d'enfants à conserver.

        Returns:
            Child chunks triés avec metadata['rerank_score'].
        """
        if not child_documents:
            return []

        limit = top_k if top_k is not None else self.final_k
        pairs = [(query, doc.page_content) for doc in child_documents]

        raw_scores = self.cross_encoder.predict(pairs)

        scored_children: list[Document] = []
        for doc, score in zip(child_documents, raw_scores):
            new_metadata = dict(doc.metadata)
            new_metadata["rerank_score"] = float(score)
            scored_children.append(
                Document(page_content=doc.page_content, metadata=new_metadata)
            )

        scored_children.sort(
            key=lambda d: d.metadata.get("rerank_score", -float("inf")),
            reverse=True,
        )
        return scored_children[:limit]

    def resolve_parents(
        self,
        scored_children: Sequence[Document],
    ) -> list[Document]:
        """Associe les Child Chunks retenus à leurs Parent Chunks complets (sans doublon).

        Args:
            scored_children: Top micro-chunks classés par le Re-ranker.

        Returns:
            Liste ordonnée de documents parents complets, dédoublonnée.
        """
        resolved_parents: list[Document] = []
        seen_parent_ids: set[str] = set()

        for child in scored_children:
            parent_id = child.metadata.get("parent_id")
            score = child.metadata.get("rerank_score", 0.0)

            if parent_id and parent_id in self.parent_store:
                if parent_id not in seen_parent_ids:
                    seen_parent_ids.add(parent_id)
                    base_parent = self.parent_store[parent_id]
                    # Copie avec enrichissement de métadonnées de pertinence
                    parent_meta = dict(base_parent.metadata)
                    parent_meta["rerank_score"] = score
                    parent_meta["matched_snippet"] = child.page_content
                    resolved_parents.append(
                        Document(
                            page_content=base_parent.page_content,
                            metadata=parent_meta,
                        )
                    )
            else:
                # Mode de secours : si aucun parent n'est enregistré, utiliser le child directement
                doc_id = child.metadata.get("chunk_index", id(child))
                if doc_id not in seen_parent_ids:
                    seen_parent_ids.add(str(doc_id))
                    resolved_parents.append(child)

        logger.info(
            "Résolution Parent-Document : %d micro-chunks enfants résolus en %d sections parentes uniques.",
            len(scored_children),
            len(resolved_parents),
        )
        return resolved_parents

    def retrieve(
        self,
        job_description: str,
        initial_k: int | None = None,
        final_k: int | None = None,
        k: int | None = None,
    ) -> list[Document]:
        """Pipeline complet de recherche : Vector Search (Enfants) -> Re-ranking -> Résolution Parents.

        Args:
            job_description: Texte complet de la fiche de poste.
            initial_k: Nombre de micro-chunks extraits lors de l'Étape 1 (défaut: 15).
            final_k: Nombre de micro-chunks retenus après re-ranking (défaut: 4).
            k: Surcharge optionnelle de final_k (compatibilité).

        Returns:
            Sections parentes complètes correspondantes, sans troncature de phrases.
        """
        query = job_description.strip()
        if not query:
            raise ValueError("La fiche de poste (job_description) ne peut pas être vide.")

        target_final_k = k if k is not None else (final_k if final_k is not None else self.final_k)
        target_initial_k = initial_k if initial_k is not None else max(self.initial_k, target_final_k)

        # Étape 1 : Fast Retrieval sur les micro-chunks enfants via ChromaDB
        logger.info(
            "Étape 1/3 [Fast Retrieval] : extraction de k=%d micro-chunks depuis ChromaDB...",
            target_initial_k,
        )
        candidate_children = self.vector_store.similarity_search(query, k=target_initial_k)

        if not candidate_children:
            logger.warning("Aucun micro-chunk candidat trouvé dans ChromaDB.")
            return []

        # Étape 2 : Precision Re-ranking via Cross-Encoder sur les micro-chunks
        logger.info(
            "Étape 2/3 [Precision Re-ranking] : réordonnancement de %d micro-chunks vers Top-%d...",
            len(candidate_children),
            target_final_k,
        )
        top_children = self.rerank_children(query=query, child_documents=candidate_children, top_k=target_final_k)

        # Étape 3 : Résolution Parent-Document (Sections complètes dédoublonnées)
        logger.info("Étape 3/3 [Parent Resolution] : extraction des sections parentes complètes...")
        return self.resolve_parents(top_children)

    def analyze(
        self,
        job_description: str,
        initial_k: int | None = None,
        final_k: int | None = None,
        k: int | None = None,
    ) -> MatchReport:
        """Exécute la recherche documentaire avancée puis génère le MatchReport structuré.

        Args:
            job_description: Texte complet de la fiche de poste.

        Returns:
            Une instance validée de ``MatchReport``.
        """
        parent_docs = self.retrieve(job_description, initial_k=initial_k, final_k=final_k, k=k)
        messages = self._build_messages(job_description, parent_docs)
        logger.info(
            "Appel du LLM structuré avec %d section(s) parente(s) complète(s).",
            len(parent_docs),
        )
        raw_output = self.structured_llm.invoke(messages)

        if isinstance(raw_output, MatchReport):
            return raw_output

        return MatchReport.model_validate(raw_output)

    def analyze_resume(
        self,
        pdf_path: str | Path,
        job_description: str,
        initial_k: int | None = None,
        final_k: int | None = None,
        k: int | None = None,
    ) -> MatchReport:
        """Méthode de bout en bout : indexation Parent-Child du CV PDF puis analyse RAG."""
        self.index_resume(pdf_path, reset=True)
        return self.analyze(job_description, initial_k=initial_k, final_k=final_k, k=k)

    @staticmethod
    def _format_context(documents: list[Document]) -> str:
        """Formate les sections parentes en contexte propre et balisé pour le LLM."""
        if not documents:
            return (
                "[Aucun extrait de CV n'a été trouvé dans la base vectorielle. "
                "Le score doit être 0 ou très faible, matching_skills vide, "
                "et toutes les compétences de la fiche dans missing_skills.]"
            )

        blocks: list[str] = []
        for index, doc in enumerate(documents, start=1):
            page = doc.metadata.get("page", "?")
            parent_idx = doc.metadata.get("parent_index", "?")
            filename = doc.metadata.get("filename", "cv.pdf")
            score = doc.metadata.get("rerank_score")
            score_tag = f" | Pertinence Re-ranker: {score:+.3f}" if score is not None else ""
            header = f"--- Section Parente #{index} | Fichier: {filename} (page {page}, section {parent_idx}){score_tag} ---"
            blocks.append(f"{header}\n{doc.page_content.strip()}")
        return "\n\n".join(blocks)

    def _build_messages(
        self,
        job_description: str,
        documents: list[Document],
    ) -> list[SystemMessage | HumanMessage]:
        """Construit les messages (System + Human) avec les sections parentes non tronquées."""
        formatted_context = self._format_context(documents)
        human_content = (
            "FICHE DE POSTE CIBLE (Job Description) :\n"
            "=========================================\n"
            f"{job_description.strip()}\n"
            "=========================================\n\n"
            "SECTIONS DU CV DU CANDIDAT (Sections parentes complètes - Seule source autorisée) :\n"
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
# Smoke test complet : Parent-Child Ingestion -> Chroma -> Re-ranking -> LLM
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
- Re-ranking Cross-Encoder
- Pydantic v2
""".strip()

    print("\n" + "=" * 70)
    print("🚀 SMOKE TEST : Parent-Document Retriever + Cross-Encoder Re-ranking")
    print("=" * 70)
    print(f"📄 CV Path             : {pdf_path}")
    print(f"🤖 LLM Model           : {settings.OLLAMA_MODEL} @ {settings.OLLAMA_BASE_URL}")
    print(f"🧠 Bi-Encoder (Stage 1): {settings.EMBEDDING_MODEL_NAME}")
    print(f"⚖️ Cross-Encoder(Stage 2): {getattr(settings, 'CROSS_ENCODER_MODEL_NAME', 'cross-encoder/ms-marco-MiniLM-L-6-v2')}")
    print(f"🎯 Retrieval Strategy  : Chroma Child ({DEFAULT_INITIAL_K}) -> Re-rank Top-{DEFAULT_FINAL_K} -> Resolve Parent")
    print("-" * 70)

    if not pdf_path.exists():
        logger.error("Le fichier CV spécifié n'existe pas : %s", pdf_path)
        sys.exit(1)

    try:
        pipeline = RAGPipeline(
            initial_k=DEFAULT_INITIAL_K,
            final_k=DEFAULT_FINAL_K,
        )

        print("\n[1/4] Ingestion Parent-Child (Sanitization + Indexation ChromaDB)...")
        num_children = pipeline.index_resume(pdf_path, reset=True)
        print(f"      ✅ {len(pipeline.parent_store)} sections parentes enregistrées.")
        print(f"      ✅ {num_children} micro-chunks enfants indexés dans ChromaDB.")

        print(f"\n[2/4] Recherche Two-Stage & Résolution des sections parentes...")
        parent_sections = pipeline.retrieve(job_description)
        for i, doc in enumerate(parent_sections, start=1):
            score = doc.metadata.get("rerank_score", 0.0)
            preview = doc.page_content.replace("\n", " ")[:80]
            print(f"      • Section #{i} [Score: {score:+.4f}] : {preview}...")

        print("\n[3/4] Inférence LLM Ollama avec MatchReport structuré...")
        report: MatchReport = pipeline.analyze(job_description)
        print("      ✅ Rapport structuré obtenu avec succès.")

        print("\n[4/4] Résultat du MatchReport :")
        print("=" * 70)
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
        print("=" * 70)

        print("\n📦 Export JSON validé Pydantic v2 :")
        print(report.model_dump_json(indent=2))
        print("\n🎉 Smoke test Parent-Document Retriever validé avec succès !")

    except (FileNotFoundError, PDFParserError) as err:
        logger.error("Erreur de parsing : %s", err)
        sys.exit(1)
    except Exception as err:
        logger.error("Erreur durant le pipeline Advanced RAG : %s", err, exc_info=True)
        sys.exit(1)
