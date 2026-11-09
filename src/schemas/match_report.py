"""Pydantic v2 schemas for the structured CV ↔ job-description analysis."""

from __future__ import annotations

from typing import List
from pydantic import BaseModel, ConfigDict, Field


class MatchReport(BaseModel):
    """Structured fit analysis between a resume and a job description.

    Produced by the local LLM via ``ChatOllama.with_structured_output(MatchReport)``.
    Every field is validated so the UI and downstream pipeline can rely on typed,
    guaranteed data without hallucinated structures.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="ignore",
    )

    match_score: int = Field(
        ...,
        ge=0,
        le=100,
        description=(
            "Score d'adéquation global entre le CV et la fiche de poste, "
            "entier de 0 (aucun recouvrement) à 100 (adéquation complète). "
            "Doit refléter uniquement les preuves factuelles présentes dans les extraits du CV."
        ),
    )
    matching_skills: List[str] = Field(
        ...,
        description=(
            "Compétences, outils, technologies ou expériences explicitement "
            "mentionnés dans les extraits du CV ET exigés ou pertinents pour le poste."
        ),
    )
    missing_skills: List[str] = Field(
        ...,
        description=(
            "Compétences, outils ou exigences requises par la fiche de poste "
            "qui sont absentes ou non vérifiables dans les extraits du CV."
        ),
    )
    summary: str = Field(
        ...,
        description=(
            "Synthèse factuelle et objective (3 à 6 phrases) de la correspondance, "
            "basée rigoureusement et exclusivement sur les extraits fournis."
        ),
    )
    recommendations: List[str] = Field(
        ...,
        description=(
            "Recommandations concrètes et actionnables pour optimiser le CV vis-à-vis "
            "de ce poste (mise en valeur de mots-clés justifiés, reformulation, points à clarifier)."
        ),
    )
