# 🎯 Smart Resume Matcher & Career Assistant

> **Système RAG 100% Local, Souverain & Privacy-First pour l'évaluation objective de candidatures.**  
> Développé en Python avec **LangChain**, **Ollama (`llama3.2`)**, **ChromaDB**, **Sentence-Transformers (MPS)**, **Pydantic v2** et **Streamlit**.

---

## 🌟 Philosophie & Points Clés

- **🔒 100% Local & Privacy-First :** Aucune donnée personnelle, aucun CV et aucune fiche de poste ne transitent vers des APIs cloud propriétaires (OpenAI, Anthropic, etc.). Vos données restent strictement sur votre machine (conformité RGPD by design).
- **⚡ Optimisation Apple Silicon (MPS) :** Accélération matérielle des embeddings via le backend `mps` (Metal Performance Shaders) de PyTorch sur Mac M1/M2/M3, avec mise en cache mémoire (`@lru_cache`).
- **🛡️ Zéro Hallucination & Sorties Structurées :** Prompting RAG strict interdisant toute invention d'expérience, forcé via le mécanisme `.with_structured_output(MatchReport)` de LangChain vers un schéma Pydantic v2 rigoureusement validé.
- **📊 Traçabilité Totale :** Chaque chunk vectorisé conserve ses métadonnées d'origine (nom du fichier, numéro de page, index du chunk) permettant de citer les preuves exactes du CV lors de l'évaluation.

---

## 🏗️ Architecture & Workflow RAG

```text
┌────────────────────────┐         ┌────────────────────────┐
│     CV Candidat        │         │     Fiche de Poste     │
│       (PDF)            │         │         (Texte)        │
└───────────┬────────────┘         └───────────┬────────────┘
            │                                  │
      [ PDFParser ]                            │
  (Extraction + Chunking                       │
   RecursiveSplitter)                          │
            │                                  │
            ▼                                  │
     [ Chunks + Meta ]                         │
            │                                  │
  [ Sentence-Transformers ]                    │
      (MPS / Local GPU)                        │
            │                                  │
            ▼                                  │
     [ ChromaDB Store ]                        │
       (Indexation)                            │
            │                                  │
            └───────────────┬──────────────────┘
                            ▼
               [ Recherche de Similarité ]
                   (Top-k = 4 Chunks)
                            │
                            ▼
               [ Prompt RAG Strict RH ]
              (Anti-hallucination, Français)
                            │
                            ▼
                 [ ChatOllama (Local) ]
                (Modèle: llama3.2 @ 0.1)
                            │
                            ▼
              [ MatchReport (Pydantic v2) ]
        ┌───────────────────┴───────────────────┐
        ▼                                       ▼
 📊 UI Web Streamlit                     📦 Export JSON
 (Score, Skills, Conseils)            (API / Intégration ATS)
```

---

## 📁 Arborescence du Projet

```text
FIRST/
├── app/
│   ├── __init__.py
│   └── main.py                     # Interface Web utilisateur Streamlit
├── src/
│   ├── __init__.py
│   ├── config.py                   # Configuration centralisée & chemins
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── embeddings_factory.py   # Factory Sentence-Transformers sur MPS/CPU
│   ├── llm/
│   │   ├── __init__.py
│   │   └── ollama_client.py        # Client ChatOllama & Structured Output
│   ├── parsing/
│   │   ├── __init__.py
│   │   └── pdf_parser.py           # Parsing PDF (pypdf) & Semantic Chunking
│   ├── rag/
│   │   ├── __init__.py
│   │   └── pipeline.py             # Orchestrateur RAG de bout en bout
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── match_report.py         # Schéma Pydantic v2 (MatchReport)
│   └── vectorstore/
│       ├── __init__.py
│       └── chromadb_manager.py     # Gestionnaire persistant ChromaDB
├── data/
│   ├── resumes/                    # Échantillons de CV (PDF)
│   └── jobs/                       # Échantillons de fiches de poste (TXT)
├── tests/
│   ├── __init__.py
│   └── test_step3.py               # Tests unitaires (Schémas, LLM, RAG)
├── chroma_db/                      # Base vectorielle locale persistante (Chroma)
├── requirements.txt                # Dépendances Python testées
└── README.md                       # Documentation du projet
```

---

## 🛠️ Prérequis Système

1. **Système d'exploitation :** macOS (Apple Silicon M1/M2/M3 recommandé) ou Linux / Windows avec GPU compatible.
2. **Python :** Version `3.11` ou `3.13`.
3. **Ollama :** Moteur LLM local installé et actif.
   - Installation : [ollama.com](https://ollama.com)
   - Téléchargement du modèle requis :
     ```bash
     ollama pull llama3.2
     ```

---

## 🚀 Guide d'Installation Rapide

### 1. Cloner le dépôt et se placer dans le projet
```bash
git clone https://github.com/votre-compte/RAG-smart-resume-matcher.git
cd RAG-smart-resume-matcher
```

### 2. Créer et activer l'environnement virtuel
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 4. Démarrer le serveur Ollama
Dans un terminal dédié :
```bash
ollama serve
```

---

## 🖥️ Utilisation & Commandes de Lancement

### Lancer l'application Web Streamlit
Depuis la racine du projet, exécutez la commande suivante :

```bash
PYTHONPATH=. streamlit run app/main.py
```
> L'application s'ouvrira automatiquement à l'adresse locale : `http://localhost:8501`

### Lancer les Tests Unitaires
Pour vérifier la conformité des schémas Pydantic, du client LLM et du pipeline :

```bash
python -m unittest tests/test_step3.py
```

### Lancer le Smoke Test CLI complet
Pour valider l'ensemble du pipeline en ligne de commande (Parsing PDF -> Indexation -> RAG -> MatchReport JSON) :

```bash
PYTHONPATH=. python src/rag/pipeline.py
```

---

## 📋 Schéma de Données (`MatchReport`)

La réponse du LLM est systématiquement validée par Pydantic v2 :

```json
{
  "match_score": 85,
  "matching_skills": [
    "Python",
    "LangChain",
    "ChromaDB",
    "Sentence-Transformers",
    "Apple Silicon (MPS)"
  ],
  "missing_skills": [
    "Docker"
  ],
  "summary": "Le candidat présente une excellente adéquation technique avec les exigences du poste d'ingénieur RAG. L'expérience avec les outils locaux et l'optimisation Apple Silicon est parfaitement démontrée. Seule la compétence Docker n'est pas explicitée dans le CV.",
  "recommendations": [
    "Mentionner explicitement les éventuels déploiements ou conteneurisations Docker réalisés.",
    "Mettre en avant les métriques de latence obtenues sur l'accélération MPS lors des projets précédents."
  ]
}
```

---

## ⚖️ Licence & Confidentialité

Projet distribué sous licence open-source MIT.  
Conçu pour préserver à 100% l'anonymat et la confidentialité des candidatures.
