# rag/sql_schemas.py
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator


def _empty_to_none(value):
    """Convertit les valeurs vides / NaN pandas en None."""
    if value is None:
        return None

    # pandas/numpy NaN: seul cas où value != value
    try:
        if value != value:
            return None
    except Exception:
        pass

    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return None

    return value


class PlayerStatRow(BaseModel):
    """Ligne normalisée de la feuille 'Données NBA'."""

    source_file: str = Field(..., min_length=1)
    source_sheet: str = Field(..., min_length=1)
    source_row_number: int = Field(..., ge=1)

    player: str = Field(..., min_length=1)
    team_code: str = Field(..., min_length=1)

    age: Optional[int] = None
    gp: Optional[int] = None
    wins: Optional[int] = None
    losses: Optional[int] = None
    minutes: Optional[float] = None

    pts: Optional[float] = None
    fgm: Optional[float] = None
    fga: Optional[float] = None
    fg_pct: Optional[float] = None

    three_pm: Optional[float] = None
    three_pa: Optional[float] = None
    three_p_pct: Optional[float] = None

    ftm: Optional[float] = None
    fta: Optional[float] = None
    ft_pct: Optional[float] = None

    oreb: Optional[float] = None
    dreb: Optional[float] = None
    reb: Optional[float] = None
    ast: Optional[float] = None
    tov: Optional[float] = None
    stl: Optional[float] = None
    blk: Optional[float] = None
    pf: Optional[float] = None

    fp: Optional[float] = None
    dd2: Optional[float] = None
    td3: Optional[float] = None
    plus_minus: Optional[float] = None

    offrtg: Optional[float] = None
    defrtg: Optional[float] = None
    netrtg: Optional[float] = None

    ast_pct: Optional[float] = None
    ast_to_ratio: Optional[float] = None
    ast_ratio: Optional[float] = None

    oreb_pct: Optional[float] = None
    dreb_pct: Optional[float] = None
    reb_pct: Optional[float] = None

    to_ratio: Optional[float] = None
    efg_pct: Optional[float] = None
    ts_pct: Optional[float] = None
    usg_pct: Optional[float] = None
    pace: Optional[float] = None
    pie: Optional[float] = None
    poss: Optional[float] = None

    @field_validator(
        "source_file", "source_sheet", "player", "team_code", mode="before"
    )
    @classmethod
    def clean_required_str(cls, value):
        value = _empty_to_none(value)
        if value is None:
            raise ValueError("Champ texte obligatoire manquant.")
        return str(value).strip()

    @field_validator(
        "age",
        "gp",
        "wins",
        "losses",
        mode="before",
    )
    @classmethod
    def clean_optional_int(cls, value):
        value = _empty_to_none(value)
        if value is None:
            return None
        return int(float(value))

    @field_validator(
        "minutes",
        "pts",
        "fgm",
        "fga",
        "fg_pct",
        "three_pm",
        "three_pa",
        "three_p_pct",
        "ftm",
        "fta",
        "ft_pct",
        "oreb",
        "dreb",
        "reb",
        "ast",
        "tov",
        "stl",
        "blk",
        "pf",
        "fp",
        "dd2",
        "td3",
        "plus_minus",
        "offrtg",
        "defrtg",
        "netrtg",
        "ast_pct",
        "ast_to_ratio",
        "ast_ratio",
        "oreb_pct",
        "dreb_pct",
        "reb_pct",
        "to_ratio",
        "efg_pct",
        "ts_pct",
        "usg_pct",
        "pace",
        "pie",
        "poss",
        mode="before",
    )
    @classmethod
    def clean_optional_float(cls, value):
        value = _empty_to_none(value)
        if value is None:
            return None
        return float(value)


class TeamRow(BaseModel):
    """Ligne normalisée de la feuille 'Equipe'."""

    source_file: str = Field(..., min_length=1)
    source_sheet: str = Field(..., min_length=1)
    source_row_number: int = Field(..., ge=1)

    team_code: str = Field(..., min_length=1)
    team_name: str = Field(..., min_length=1)

    @field_validator(
        "source_file", "source_sheet", "team_code", "team_name", mode="before"
    )
    @classmethod
    def clean_required_str(cls, value):
        value = _empty_to_none(value)
        if value is None:
            raise ValueError("Champ texte obligatoire manquant.")
        return str(value).strip()


class MetricDictionaryRow(BaseModel):
    """Ligne normalisée de la feuille 'Dictionnaire des données'."""

    source_file: str = Field(..., min_length=1)
    source_sheet: str = Field(..., min_length=1)
    source_row_number: int = Field(..., ge=1)

    metric_code: str = Field(..., min_length=1)
    metric_description: str = Field(..., min_length=1)

    @field_validator(
        "source_file",
        "source_sheet",
        "metric_code",
        "metric_description",
        mode="before",
    )
    @classmethod
    def clean_required_str(cls, value):
        value = _empty_to_none(value)
        if value is None:
            raise ValueError("Champ texte obligatoire manquant.")
        return str(value).strip()


class SQLToolResult(BaseModel):
    """Résultat structuré renvoyé par le tool SQL."""

    question: str = Field(..., min_length=1)
    sql_query: str = Field(..., min_length=1)
    rows: list[dict] = Field(default_factory=list)
    row_count: int = Field(..., ge=0)
    notes: Optional[str] = None

    @field_validator("question", "sql_query", mode="before")
    @classmethod
    def clean_required_str(cls, value):
        value = _empty_to_none(value)
        if value is None:
            raise ValueError("Champ texte obligatoire manquant.")
        return str(value).strip()

    @field_validator("notes", mode="before")
    @classmethod
    def clean_optional_notes(cls, value):
        value = _empty_to_none(value)
        if value is None:
            return None
        return str(value).strip()
