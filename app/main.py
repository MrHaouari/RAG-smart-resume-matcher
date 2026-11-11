"""Application Web Streamlit - Smart Resume Matcher & Career Assistant.

Interface utilisateur locale et confidentielle (Privacy-First) pour l'analyse
d'adéquation entre un CV (PDF) et une fiche de poste via RAG local.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import httpx
import streamlit as st

from src.config import JOBS_DIR, settings
from src.parsing.pdf_parser import PDFParser, PDFParserError
from src.rag.pipeline import DEFAULT_RETRIEVER_K, RAGPipeline
from src.schemas.match_report import MatchReport
from src.vectorstore.chromadb_manager import VectorStoreManager

# Configuration du logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Configuration de la page Streamlit
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Smart Resume Matcher | Assistant Carrière",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Injection de styles CSS pour une interface soignée
st.markdown(
    """
    <style>
    /* Titres */
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
        opacity: 0.85;
    }
    .badge-privacy {
        display: inline-block;
        padding: 0.3rem 0.75rem;
        font-size: 0.82rem;
        font-weight: 700;
        background-color: #DCFCE7 !important;
        color: #14532D !important;
        border: 1px solid #86EFAC !important;
        border-radius: 9999px;
        margin-bottom: 1rem;
    }

    /* Compétences validées - Contraste élevé (Mode Clair par défaut) */
    .skill-card-match {
        background-color: #DCFCE7 !important;
        color: #14532D !important;
        border: 1px solid #86EFAC !important;
        border-left: 6px solid #16A34A !important;
        padding: 0.75rem 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.6rem;
        font-size: 0.95rem;
        font-weight: 600;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    .skill-card-match, .skill-card-match strong, .skill-card-match span {
        color: #14532D !important;
    }

    /* Compétences manquantes - Contraste élevé (Mode Clair par défaut) */
    .skill-card-missing {
        background-color: #FEE2E2 !important;
        color: #7F1D1D !important;
        border: 1px solid #FCA5A5 !important;
        border-left: 6px solid #DC2626 !important;
        padding: 0.75rem 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.6rem;
        font-size: 0.95rem;
        font-weight: 600;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    .skill-card-missing, .skill-card-missing strong, .skill-card-missing span {
        color: #7F1D1D !important;
    }

    /* Recommandations - Contraste élevé (Mode Clair par défaut) */
    .recommendation-box {
        background-color: #F1F5F9 !important;
        color: #0F172A !important;
        border: 1px solid #CBD5E1 !important;
        border-left: 6px solid #2563EB !important;
        border-radius: 0.5rem;
        padding: 0.9rem 1.2rem;
        margin-bottom: 0.75rem;
        font-size: 0.95rem;
        line-height: 1.55;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }
    .recommendation-box strong {
        color: #1D4ED8 !important;
        font-weight: 700;
    }
    .recommendation-box, .recommendation-box span, .recommendation-box p {
        color: #0F172A !important;
    }

    /* Support Mode Sombre (Dark Mode) : Thème Streamlit ou Système OS */
    @media (prefers-color-scheme: dark) {
        .badge-privacy {
            background-color: #064E3B !important;
            color: #A7F3D0 !important;
            border-color: #059669 !important;
        }
        .skill-card-match {
            background-color: #064E3B !important;
            color: #ECFDF5 !important;
            border: 1px solid #059669 !important;
            border-left: 6px solid #34D399 !important;
        }
        .skill-card-match, .skill-card-match strong, .skill-card-match span {
            color: #ECFDF5 !important;
        }
        .skill-card-missing {
            background-color: #7F1D1D !important;
            color: #FEF2F2 !important;
            border: 1px solid #B91C1C !important;
            border-left: 6px solid #F87171 !important;
        }
        .skill-card-missing, .skill-card-missing strong, .skill-card-missing span {
            color: #FEF2F2 !important;
        }
        .recommendation-box {
            background-color: #1E293B !important;
            color: #F8FAFC !important;
            border: 1px solid #334155 !important;
            border-left: 6px solid #60A5FA !important;
        }
        .recommendation-box strong {
            color: #93C5FD !important;
        }
        .recommendation-box, .recommendation-box span, .recommendation-box p {
            color: #F8FAFC !important;
        }
    }

    /* Règle explicite pour Streamlit Dark Theme */
    [data-theme="dark"] .badge-privacy {
        background-color: #064E3B !important;
        color: #A7F3D0 !important;
        border-color: #059669 !important;
    }
    [data-theme="dark"] .skill-card-match {
        background-color: #064E3B !important;
        color: #ECFDF5 !important;
        border: 1px solid #059669 !important;
        border-left: 6px solid #34D399 !important;
    }
    [data-theme="dark"] .skill-card-match,
    [data-theme="dark"] .skill-card-match strong,
    [data-theme="dark"] .skill-card-match span {
        color: #ECFDF5 !important;
    }
    [data-theme="dark"] .skill-card-missing {
        background-color: #7F1D1D !important;
        color: #FEF2F2 !important;
        border: 1px solid #B91C1C !important;
        border-left: 6px solid #F87171 !important;
    }
    [data-theme="dark"] .skill-card-missing,
    [data-theme="dark"] .skill-card-missing strong,
    [data-theme="dark"] .skill-card-missing span {
        color: #FEF2F2 !important;
    }
    [data-theme="dark"] .recommendation-box {
        background-color: #1E293B !important;
        color: #F8FAFC !important;
        border: 1px solid #334155 !important;
        border-left: 6px solid #60A5FA !important;
    }
    [data-theme="dark"] .recommendation-box strong {
        color: #93C5FD !important;
    }
    [data-theme="dark"] .recommendation-box,
    [data-theme="dark"] .recommendation-box span,
    [data-theme="dark"] .recommendation-box p {
        color: #F8FAFC !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def check_ollama_status(base_url: str) -> bool:
    """Vérifie si le démon Ollama répond sur l'URL configurée."""
    try:
        response = httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=1.5)
        return response.status_code == 200
    except Exception:
        return False


def load_sample_job_description() -> str:
    """Charge l'exemple de fiche de poste s'il existe."""
    sample_path = JOBS_DIR / "sample_job.txt"
    if sample_path.exists():
        return sample_path.read_text(encoding="utf-8")
    return (
        "Développeur Python / RAG — CDI — Paris (hybride)\n\n"
        "Compétences exigées :\n"
        "- Python avancé\n"
        "- LangChain & ChromaDB\n"
        "- LLMs locaux (Ollama, llama3.2)\n"
        "- Sentence-Transformers\n"
        "- Expérience Apple Silicon (Mac M1/M2)\n\n"
        "Compétences souhaitées :\n"
        "- Streamlit\n"
        "- Pydantic v2\n"
        "- Docker\n"
    )


# -----------------------------------------------------------------------------
# Sidebar : Configuration et Statut Système
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Architecture & Statut")
    st.markdown(
        '<span class="badge-privacy">🔒 100% Local & Privacy-First</span>',
        unsafe_allow_html=True,
    )

    ollama_online = check_ollama_status(settings.OLLAMA_BASE_URL)
    if ollama_online:
        st.success(f"🟢 Ollama connecté : `{settings.OLLAMA_MODEL}`")
    else:
        st.error("🔴 Ollama injoignable")
        st.caption(
            f"Vérifiez qu'Ollama est actif sur `{settings.OLLAMA_BASE_URL}`.\n\n"
            "Commande : `ollama serve`"
        )

    st.divider()

    st.markdown("**Paramètres du pipeline :**")
    retriever_k = st.slider(
        "Nombre d'extraits CV (k chunks)",
        min_value=2,
        max_value=8,
        value=DEFAULT_RETRIEVER_K,
        help="Nombre de segments de CV les plus pertinents injectés dans le contexte RAG.",
    )

    st.markdown(f"• **LLM :** `{settings.OLLAMA_MODEL}`")
    st.markdown("• **Bi-Encoder :** `all-MiniLM-L6-v2` (MPS)")
    st.markdown(f"• **Cross-Encoder :** `{getattr(settings, 'CROSS_ENCODER_MODEL_NAME', 'ms-marco-MiniLM-L-6-v2').split('/')[-1]}`")
    st.markdown("• **Vector Store :** `ChromaDB` (local)")

    st.divider()
    st.caption(
        "Aucune donnée personnelle n'est envoyée vers un cloud tiers. "
        "L'ensemble du parsing, des vecteurs et de l'inférence tourne en local."
    )


# -----------------------------------------------------------------------------
# En-tête principal
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">🎯 Smart Resume Matcher & Career Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Évaluation objective et sans hallucination de l\'adéquation entre votre CV et une offre d\'emploi via RAG local.</div>',
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Formulaire d'entrée : CV et Fiche de poste
# -----------------------------------------------------------------------------
col_cv, col_job = st.columns(2, gap="medium")

with col_cv:
    st.subheader("📄 1. Import du CV (PDF)")
    uploaded_file = st.file_uploader(
        "Sélectionnez un fichier CV au format PDF",
        type=["pdf"],
        help="Le PDF sera analysé localement et découpé en chunks sémantiques.",
    )

    if uploaded_file is not None:
        st.info(f"Fichier sélectionné : **{uploaded_file.name}** ({uploaded_file.size / 1024:.1f} Ko)")

with col_job:
    st.subheader("📋 2. Fiche de Poste")
    
    col_btn, _ = st.columns([1, 1])
    with col_btn:
        if st.button("📝 Charger un exemple de poste", use_container_width=True):
            st.session_state["job_description_input"] = load_sample_job_description()

    default_text = st.session_state.get("job_description_input", "")
    job_description = st.text_area(
        "Collez ici la description complète du poste :",
        value=default_text,
        height=220,
        placeholder="Ex: Développeur Python senior, maîtrise de FastAPI, Docker, SQL...",
        key="job_description_input",
    )

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Exécution du Pipeline RAG
# -----------------------------------------------------------------------------
analyze_btn = st.button("🚀 Lancer l'analyse d'adéquation", type="primary", use_container_width=True)

if analyze_btn:
    # Validations préalables
    if not uploaded_file:
        st.warning("⚠️ Veuillez importer un CV au format PDF avant de lancer l'analyse.")
    elif not job_description.strip():
        st.warning("⚠️ Veuillez renseigner la fiche de poste à comparer.")
    elif not ollama_online:
        st.error(
            f"❌ Impossible de joindre le serveur Ollama sur {settings.OLLAMA_BASE_URL}.\n\n"
            "Veuillez démarrer Ollama dans votre terminal avec `ollama serve` et vérifier la présence du modèle avec `ollama pull llama3.2`."
        )
    else:
        temp_pdf_path = None
        try:
            # 1. Gestion sécurisée du fichier temporaire pour le PDF uploadé
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_pdf_path = Path(tmp_file.name)

            progress_bar = st.progress(0, text="Initialisation du pipeline RAG...")

            # 2 & 3. Ingestion Parent-Child (Sanitization + Indexation micro-chunks)
            progress_bar.progress(30, text="Sanitization et découpage Parent-Child (Sections complètes & Micro-chunks)...")
            pipeline = RAGPipeline(retriever_k=retriever_k)
            num_children = pipeline.index_resume(temp_pdf_path, reset=True)

            if num_children == 0:
                raise PDFParserError("Aucun texte exploitable n'a été extrait du document PDF.")

            # 4. Exécution du RAG (Re-ranking Cross-Encoder -> Résolution Parent) et inférence LLM
            progress_bar.progress(70, text="Recherche vectorielle (Child Chunks) + Re-ranking Cross-Encoder -> Résolution Parents...")
            report: MatchReport = pipeline.analyze(job_description)

            progress_bar.progress(100, text="Analyse terminée avec succès !")

            # -----------------------------------------------------------------
            # Affichage soigné des résultats
            # -----------------------------------------------------------------
            st.divider()
            st.markdown("## 📊 Résultats de l'analyse d'adéquation")

            # Métrique principale et indicateur visuel
            score = report.match_score
            col_metric, col_status = st.columns([1, 3])

            with col_metric:
                st.metric(
                    label="Score d'adéquation global",
                    value=f"{score} / 100",
                    delta=f"{score - 50}% vs médiane" if score != 50 else None,
                )

            with col_status:
                if score >= 80:
                    st.success("🌟 **Excellente adéquation :** Le profil répond très fidèlement aux exigences majeures du poste.")
                elif score >= 60:
                    st.info("👍 **Bonne adéquation :** Profil solide présentant les bases requises, avec quelques axes de progression.")
                elif score >= 40:
                    st.warning("⚠️ **Adéquation partielle :** Plusieurs compétences clés de l'offre ne sont pas prouvées dans le CV.")
                else:
                    st.error("❌ **Faible adéquation :** Écart significatif entre les exigences du poste et les éléments mentionnés dans le CV.")

            st.markdown("<br>", unsafe_allow_html=True)

            # Colonnes Compétences : Validées vs Manquantes
            col_match, col_miss = st.columns(2, gap="large")

            with col_match:
                st.markdown(f"### ✅ Compétences validées ({len(report.matching_skills)})")
                st.caption("Éléments explicitement mentionnés dans le CV et demandés par le poste :")
                if report.matching_skills:
                    for skill in report.matching_skills:
                        st.markdown(
                            f'<div class="skill-card-match">✔ <strong>{skill}</strong></div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.write("_Aucune correspondance directe identifiée dans les extraits._")

            with col_miss:
                st.markdown(f"### ❌ Compétences manquantes ({len(report.missing_skills)})")
                st.caption("Exigences du poste non justifiées dans les extraits du CV :")
                if report.missing_skills:
                    for skill in report.missing_skills:
                        st.markdown(
                            f'<div class="skill-card-missing">✖ <strong>{skill}</strong></div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.write("_Aucune compétence manquante majeure détectée._")

            st.markdown("<br>", unsafe_allow_html=True)

            # Synthèse RH factuelle
            st.markdown("### 📝 Synthèse de l'évaluation")
            st.info(report.summary)

            # Recommandations concrètes
            st.markdown("### 💡 Conseils & Recommandations d'optimisation")
            if report.recommendations:
                for idx, rec in enumerate(report.recommendations, start=1):
                    st.markdown(
                        f"""
                        <div class="recommendation-box">
                            <strong>{idx}.</strong> {rec}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.write("_Aucune recommandation spécifique requise._")

            # Expander de transparence : inspection des chunks RAG
            with st.expander("🔍 Voir les extraits du CV utilisés pour cette analyse (Preuves RAG & Re-ranking)"):
                retrieved_docs = pipeline.retrieve(job_description, k=retriever_k)
                for i, doc in enumerate(retrieved_docs, start=1):
                    page = doc.metadata.get("page", "?")
                    chunk_idx = doc.metadata.get("chunk_index", "?")
                    score = doc.metadata.get("rerank_score")
                    score_info = f" — Score Cross-Encoder: {score:+.3f}" if score is not None else ""
                    st.markdown(f"**Extrait #{i}** *(Page {page}, Chunk #{chunk_idx}{score_info})*")
                    st.code(doc.page_content, language="markdown")

            # Expander JSON brut
            with st.expander("📦 Données JSON brutes (MatchReport validé Pydantic v2)"):
                st.json(report.model_dump())

        except PDFParserError as parse_err:
            st.error(f"Erreur de lecture du PDF : {parse_err}")
            logger.error("Erreur PDFParser : %s", parse_err)
        except (httpx.ConnectError, ConnectionRefusedError) as conn_err:
            st.error(
                f"Erreur de connexion à Ollama : impossible d'atteindre `{settings.OLLAMA_BASE_URL}`. "
                "Assurez-vous qu'Ollama est en cours d'exécution avec `ollama serve`."
            )
            logger.error("Erreur connexion Ollama : %s", conn_err)
        except Exception as e:
            st.error(f"Une erreur imprévue est survenue : {e}")
            logger.error("Erreur inattendue dans l'application : %s", e, exc_info=True)
        finally:
            # Nettoyage systématique du fichier temporaire
            if temp_pdf_path and temp_pdf_path.exists():
                try:
                    os.unlink(temp_pdf_path)
                    logger.info("Fichier temporaire supprimé : %s", temp_pdf_path)
                except Exception as clean_err:
                    logger.warning("Échec de la suppression du fichier temporaire : %s", clean_err)
