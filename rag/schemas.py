# rag/schemas.py
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator


class ParsedDocument(BaseModel):
    """Document chargé et normalisé avant chunking."""

    page_content: str = Field(..., min_length=1)
    source: str = Field(..., min_length=1)
    filename: str = Field(..., min_length=1)
    category: str = Field(..., min_length=1)
    full_path: str = Field(..., min_length=1)
    sheet: Optional[str] = None

    @field_validator("page_content", "source", "filename", "category", "full_path")
    @classmethod
    def strip_required_strings(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Le champ ne peut pas être vide.")
        return value

    @field_validator("sheet")
    @classmethod
    def strip_optional_sheet(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        return value or None


class Chunk(BaseModel):
    """Chunk prêt à être indexé dans la base vectorielle."""

    id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    source: str = Field(..., min_length=1)
    filename: str = Field(..., min_length=1)
    category: str = Field(..., min_length=1)
    full_path: str = Field(..., min_length=1)
    sheet: Optional[str] = None
    chunk_id_in_doc: int = Field(..., ge=0)
    start_index: int = Field(..., ge=0)

    @field_validator("id", "text", "source", "filename", "category", "full_path")
    @classmethod
    def strip_required_strings(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Le champ ne peut pas être vide.")
        return value

    @field_validator("sheet")
    @classmethod
    def strip_optional_sheet(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        return value or None


class UserQuestion(BaseModel):
    """Question reçue par le système au runtime."""

    question: str = Field(..., min_length=1)
    lang: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    top_k: Optional[int] = Field(default=None, ge=1)

    @field_validator("question")
    @classmethod
    def strip_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("La question ne peut pas être vide.")
        return value

    @field_validator("lang", "user_id", "session_id")
    @classmethod
    def strip_optional_strings(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        return value or None


class RetrievedChunk(BaseModel):
    """Résultat individuel renvoyé par le retrieval."""

    score: float
    raw_score: float
    text: str = Field(..., min_length=1)
    source: str = Field(..., min_length=1)
    filename: str = Field(..., min_length=1)
    category: str = Field(..., min_length=1)
    full_path: str = Field(..., min_length=1)
    sheet: Optional[str] = None

    @field_validator("text", "source", "filename", "category", "full_path")
    @classmethod
    def strip_required_strings(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Le champ ne peut pas être vide.")
        return value

    @field_validator("sheet")
    @classmethod
    def strip_optional_sheet(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        return value or None


class RetrievedContext(BaseModel):
    """Contexte complet récupéré pour une question."""

    question: str = Field(..., min_length=1)
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    k: int = Field(..., ge=1)
    context_str: str = ""

    @field_validator("question")
    @classmethod
    def strip_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("La question ne peut pas être vide.")
        return value


class FinalAnswer(BaseModel):
    """Réponse finale structurée du système."""

    answer: str = Field(..., min_length=1)
    sources: list[str] = Field(default_factory=list)
    used_retrieval: bool
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    notes: Optional[str] = None

    @field_validator("answer")
    @classmethod
    def strip_answer(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("La réponse ne peut pas être vide.")
        return value

    @field_validator("sources")
    @classmethod
    def clean_sources(cls, value: list[str]) -> list[str]:
        cleaned = []
        for item in value:
            item = item.strip()
            if item:
                cleaned.append(item)
        return cleaned

    @field_validator("notes")
    @classmethod
    def strip_optional_notes(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        return value or None
