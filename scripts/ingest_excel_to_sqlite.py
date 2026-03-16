from __future__ import annotations

import argparse
import logging
import re
import sys
import unicodedata
from datetime import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.sql_db import (
    DB_PATH,
    get_connection,
    initialize_database,
    insert_metric_dictionary,
    insert_player_stats,
    insert_teams,
    quick_summary,
)
from rag.sql_schemas import MetricDictionaryRow, PlayerStatRow, TeamRow

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# -----------------------------
# Helpers colonnes / nettoyage
# -----------------------------
def canonicalize_column_name(col: object) -> str:
    """Normalise un nom de colonne Excel vers un identifiant stable."""
    if col is None:
        return ""

    # Cas observé possible pour 3PM lu comme heure
    if isinstance(col, time):
        if col.hour == 15 and col.minute == 0:
            return "three_pm"
        return str(col)

    s = str(col).strip()

    # Cas observés / fréquents
    if s in {"15:00", "15:00:00"}:
        return "three_pm"
    if "+/-" in s or "±" in s:
        return "plus_minus"

    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = s.replace("%", "_pct")
    s = s.replace("+/-", "plus_minus")
    s = s.replace("/", "_")
    s = s.replace("-", "_")
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")

    mapping = {
        "player": "player",
        "team": "team_code",
        "team_code": "team_code",
        "age": "age",
        "gp": "gp",
        "w": "wins",
        "wins": "wins",
        "l": "losses",
        "losses": "losses",
        "min": "minutes",
        "minutes": "minutes",
        "pts": "pts",
        "fgm": "fgm",
        "fga": "fga",
        "fg_pct": "fg_pct",
        "3pm": "three_pm",
        "three_pm": "three_pm",
        "3pa": "three_pa",
        "three_pa": "three_pa",
        "3p_pct": "three_p_pct",
        "three_p_pct": "three_p_pct",
        "ftm": "ftm",
        "fta": "fta",
        "ft_pct": "ft_pct",
        "oreb": "oreb",
        "dreb": "dreb",
        "reb": "reb",
        "ast": "ast",
        "tov": "tov",
        "stl": "stl",
        "blk": "blk",
        "pf": "pf",
        "fp": "fp",
        "dd2": "dd2",
        "td3": "td3",
        "plus_minus": "plus_minus",
        "offrtg": "offrtg",
        "defrtg": "defrtg",
        "netrtg": "netrtg",
        "ast_pct": "ast_pct",
        "ast_to": "ast_to_ratio",
        "ast_to_ratio": "ast_to_ratio",
        "ast_ratio": "ast_ratio",
        "oreb_pct": "oreb_pct",
        "dreb_pct": "dreb_pct",
        "reb_pct": "reb_pct",
        "to_ratio": "to_ratio",
        "efg_pct": "efg_pct",
        "ts_pct": "ts_pct",
        "usg_pct": "usg_pct",
        "pace": "pace",
        "pie": "pie",
        "poss": "poss",
        # teams
        "code": "team_code",
        "abbreviation": "team_code",
        "team_abbreviation": "team_code",
        "name": "team_name",
        "team_name": "team_name",
        "full_name": "team_name",
        # dictionary
        "metric": "metric_code",
        "metric_code": "metric_code",
        "variable": "metric_code",
        "stat": "metric_code",
        "description": "metric_description",
        "definition": "metric_description",
        "metric_description": "metric_description",
    }
    return mapping.get(s, s)


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Renomme toutes les colonnes via canonicalisation."""
    renamed = {}
    seen = set()

    for col in df.columns:
        new_name = canonicalize_column_name(col)
        if not new_name:
            continue

        # évite les collisions silencieuses
        base_name = new_name
        suffix = 2
        while new_name in seen:
            new_name = f"{base_name}_{suffix}"
            suffix += 1

        renamed[col] = new_name
        seen.add(new_name)

    return df.rename(columns=renamed)


def drop_empty_rows_and_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime lignes / colonnes entièrement vides."""
    return df.dropna(axis=0, how="all").dropna(axis=1, how="all")


def first_non_null(series: pd.Series):
    for value in series:
        if pd.notna(value) and str(value).strip() != "":
            return value
    return None


# -----------------------------
# Données NBA
# -----------------------------
def load_players_stats_rows(excel_path: Path) -> list[PlayerStatRow]:
    """
    Charge la feuille 'Données NBA'.

    Important:
    - la vraie ligne d'en-tête est la ligne 2
    - on utilise donc header=1
    """
    logger.info("Chargement de la feuille 'Données NBA'...")
    df = pd.read_excel(excel_path, sheet_name="Données NBA", header=1)
    df = drop_empty_rows_and_cols(df)
    df = rename_columns(df)

    logger.info("Colonnes détectées pour 'Données NBA': %s", list(df.columns))

    # On garde surtout les lignes avec un joueur
    if "player" not in df.columns:
        raise ValueError(
            "La colonne 'player' n'a pas pu être détectée dans 'Données NBA'."
        )
    if "team_code" not in df.columns:
        raise ValueError(
            "La colonne 'team_code' n'a pas pu être détectée dans 'Données NBA'."
        )

    df = df[df["player"].notna()].copy()
    df = df.reset_index(drop=True)

    rows: list[PlayerStatRow] = []
    for i, row in df.iterrows():
        source_row_number = i + 3  # header Excel = ligne 2, données à partir de ligne 3

        payload = {
            "source_file": excel_path.name,
            "source_sheet": "Données NBA",
            "source_row_number": source_row_number,
            "player": row.get("player"),
            "team_code": row.get("team_code"),
            "age": row.get("age"),
            "gp": row.get("gp"),
            "wins": row.get("wins"),
            "losses": row.get("losses"),
            "minutes": row.get("minutes"),
            "pts": row.get("pts"),
            "fgm": row.get("fgm"),
            "fga": row.get("fga"),
            "fg_pct": row.get("fg_pct"),
            "three_pm": row.get("three_pm"),
            "three_pa": row.get("three_pa"),
            "three_p_pct": row.get("three_p_pct"),
            "ftm": row.get("ftm"),
            "fta": row.get("fta"),
            "ft_pct": row.get("ft_pct"),
            "oreb": row.get("oreb"),
            "dreb": row.get("dreb"),
            "reb": row.get("reb"),
            "ast": row.get("ast"),
            "tov": row.get("tov"),
            "stl": row.get("stl"),
            "blk": row.get("blk"),
            "pf": row.get("pf"),
            "fp": row.get("fp"),
            "dd2": row.get("dd2"),
            "td3": row.get("td3"),
            "plus_minus": row.get("plus_minus"),
            "offrtg": row.get("offrtg"),
            "defrtg": row.get("defrtg"),
            "netrtg": row.get("netrtg"),
            "ast_pct": row.get("ast_pct"),
            "ast_to_ratio": row.get("ast_to_ratio"),
            "ast_ratio": row.get("ast_ratio"),
            "oreb_pct": row.get("oreb_pct"),
            "dreb_pct": row.get("dreb_pct"),
            "reb_pct": row.get("reb_pct"),
            "to_ratio": row.get("to_ratio"),
            "efg_pct": row.get("efg_pct"),
            "ts_pct": row.get("ts_pct"),
            "usg_pct": row.get("usg_pct"),
            "pace": row.get("pace"),
            "pie": row.get("pie"),
            "poss": row.get("poss"),
        }

        try:
            rows.append(PlayerStatRow(**payload))
        except Exception as e:
            logger.warning(
                "Ligne ignorée dans 'Données NBA' (row=%s, player=%s): %s",
                source_row_number,
                payload.get("player"),
                e,
            )

    logger.info("Lignes valides players_stats: %s", len(rows))
    return rows


# -----------------------------
# Equipes
# -----------------------------
def load_team_rows(excel_path: Path) -> list[TeamRow]:
    """Charge la feuille 'Equipe' avec heuristique légère."""
    logger.info("Chargement de la feuille 'Equipe'...")
    df = pd.read_excel(excel_path, sheet_name="Equipe")
    df = drop_empty_rows_and_cols(df)
    df = rename_columns(df)

    logger.info("Colonnes détectées pour 'Equipe': %s", list(df.columns))

    code_col = "team_code" if "team_code" in df.columns else None
    name_col = "team_name" if "team_name" in df.columns else None

    if code_col is None or name_col is None:
        # fallback: on prend les 2 premières colonnes non vides
        if len(df.columns) >= 2:
            code_col = df.columns[0]
            name_col = df.columns[1]
            logger.warning(
                "Colonnes team_code/team_name non détectées clairement. "
                "Fallback sur: %s / %s",
                code_col,
                name_col,
            )
        else:
            logger.warning("Feuille 'Equipe' ignorée: colonnes insuffisantes.")
            return []

    df = df[[code_col, name_col]].copy()
    df = df.dropna(how="all").reset_index(drop=True)

    rows: list[TeamRow] = []
    for i, row in df.iterrows():
        source_row_number = i + 2
        try:
            rows.append(
                TeamRow(
                    source_file=excel_path.name,
                    source_sheet="Equipe",
                    source_row_number=source_row_number,
                    team_code=row.get(code_col),
                    team_name=row.get(name_col),
                )
            )
        except Exception as e:
            logger.warning(
                "Ligne ignorée dans 'Equipe' (row=%s): %s",
                source_row_number,
                e,
            )

    logger.info("Lignes valides teams: %s", len(rows))
    return rows


# -----------------------------
# Dictionnaire
# -----------------------------
def load_metric_dictionary_rows(excel_path: Path) -> list[MetricDictionaryRow]:
    """Charge la feuille 'Dictionnaire des données' avec heuristique légère."""
    logger.info("Chargement de la feuille 'Dictionnaire des données'...")
    df = pd.read_excel(excel_path, sheet_name="Dictionnaire des données")
    df = drop_empty_rows_and_cols(df)
    df = rename_columns(df)

    logger.info(
        "Colonnes détectées pour 'Dictionnaire des données': %s", list(df.columns)
    )

    code_col = "metric_code" if "metric_code" in df.columns else None
    desc_col = "metric_description" if "metric_description" in df.columns else None

    if code_col is None or desc_col is None:
        if len(df.columns) >= 2:
            code_col = df.columns[0]
            desc_col = df.columns[1]
            logger.warning(
                "Colonnes metric_code/metric_description non détectées clairement. "
                "Fallback sur: %s / %s",
                code_col,
                desc_col,
            )
        else:
            logger.warning(
                "Feuille 'Dictionnaire des données' ignorée: colonnes insuffisantes."
            )
            return []

    df = df[[code_col, desc_col]].copy()
    df = df.dropna(how="all").reset_index(drop=True)

    rows: list[MetricDictionaryRow] = []
    for i, row in df.iterrows():
        source_row_number = i + 2
        try:
            rows.append(
                MetricDictionaryRow(
                    source_file=excel_path.name,
                    source_sheet="Dictionnaire des données",
                    source_row_number=source_row_number,
                    metric_code=row.get(code_col),
                    metric_description=row.get(desc_col),
                )
            )
        except Exception as e:
            logger.warning(
                "Ligne ignorée dans 'Dictionnaire des données' (row=%s): %s",
                source_row_number,
                e,
            )

    logger.info("Lignes valides metric_dictionary: %s", len(rows))
    return rows


# -----------------------------
# Main
# -----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--excel-path",
        type=str,
        default="inputs/regular NBA.xlsx",
        help="Chemin vers le fichier Excel source.",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=str(DB_PATH),
        help="Chemin vers la base SQLite cible.",
    )
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="Conserve les tables existantes. Attention: risque de doublons dans players_stats.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    excel_path = Path(args.excel_path)
    db_path = Path(args.db_path)
    reset = not args.no_reset

    if not excel_path.exists():
        logger.error("Fichier Excel introuvable: %s", excel_path)
        return 1

    logger.info("Initialisation de la base SQLite: %s", db_path)
    initialize_database(db_path=db_path, reset=reset)

    player_rows = load_players_stats_rows(excel_path)
    team_rows = load_team_rows(excel_path)
    metric_rows = load_metric_dictionary_rows(excel_path)

    conn = get_connection(db_path)
    try:
        n_players = insert_player_stats(conn, player_rows)
        n_teams = insert_teams(conn, team_rows)
        n_metrics = insert_metric_dictionary(conn, metric_rows)
    finally:
        conn.close()

    summary = quick_summary(db_path)
    logger.info(
        "Ingestion terminée. Insertions: players=%s, teams=%s, metrics=%s",
        n_players,
        n_teams,
        n_metrics,
    )
    logger.info("Résumé DB: %s", summary)

    print("OK - ingestion terminée")
    print(summary)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
