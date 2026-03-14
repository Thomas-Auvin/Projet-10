# rag/sql_db.py
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

from rag.sql_schemas import MetricDictionaryRow, PlayerStatRow, TeamRow

DB_PATH = Path("data") / "nba.sqlite"


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Ouvre une connexion SQLite en créant le dossier si nécessaire."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def drop_tables(conn: sqlite3.Connection) -> None:
    """Supprime les tables si elles existent."""
    conn.executescript(
        """
        DROP TABLE IF EXISTS players_stats;
        DROP TABLE IF EXISTS teams;
        DROP TABLE IF EXISTS metric_dictionary;
        """
    )
    conn.commit()


def create_tables(conn: sqlite3.Connection) -> None:
    """Crée les tables SQLite du bloc 5."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS players_stats (
            row_id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_file TEXT NOT NULL,
            source_sheet TEXT NOT NULL,
            source_row_number INTEGER NOT NULL,

            player TEXT NOT NULL,
            team_code TEXT NOT NULL,
            age INTEGER,
            gp INTEGER,
            wins INTEGER,
            losses INTEGER,
            minutes REAL,

            pts REAL,
            fgm REAL,
            fga REAL,
            fg_pct REAL,

            three_pm REAL,
            three_pa REAL,
            three_p_pct REAL,

            ftm REAL,
            fta REAL,
            ft_pct REAL,

            oreb REAL,
            dreb REAL,
            reb REAL,
            ast REAL,
            tov REAL,
            stl REAL,
            blk REAL,
            pf REAL,

            fp REAL,
            dd2 REAL,
            td3 REAL,
            plus_minus REAL,

            offrtg REAL,
            defrtg REAL,
            netrtg REAL,

            ast_pct REAL,
            ast_to_ratio REAL,
            ast_ratio REAL,

            oreb_pct REAL,
            dreb_pct REAL,
            reb_pct REAL,

            to_ratio REAL,
            efg_pct REAL,
            ts_pct REAL,
            usg_pct REAL,
            pace REAL,
            pie REAL,
            poss REAL
        );

        CREATE TABLE IF NOT EXISTS teams (
            team_code TEXT PRIMARY KEY,
            team_name TEXT NOT NULL,
            source_file TEXT NOT NULL,
            source_sheet TEXT NOT NULL,
            source_row_number INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS metric_dictionary (
            metric_code TEXT PRIMARY KEY,
            metric_description TEXT NOT NULL,
            source_file TEXT NOT NULL,
            source_sheet TEXT NOT NULL,
            source_row_number INTEGER NOT NULL
        );
        """
    )
    conn.commit()


def create_indexes(conn: sqlite3.Connection) -> None:
    """Crée les index utiles pour les requêtes fréquentes."""
    conn.executescript(
        """
        CREATE INDEX IF NOT EXISTS idx_players_stats_player
        ON players_stats(player);

        CREATE INDEX IF NOT EXISTS idx_players_stats_team_code
        ON players_stats(team_code);

        CREATE INDEX IF NOT EXISTS idx_players_stats_pts
        ON players_stats(pts);

        CREATE INDEX IF NOT EXISTS idx_players_stats_three_p_pct
        ON players_stats(three_p_pct);

        CREATE INDEX IF NOT EXISTS idx_players_stats_age
        ON players_stats(age);

        CREATE INDEX IF NOT EXISTS idx_metric_dictionary_code
        ON metric_dictionary(metric_code);
        """
    )
    conn.commit()


def initialize_database(
    db_path: Path = DB_PATH,
    reset: bool = False,
) -> None:
    """Initialise la base SQLite, avec option de reset complet."""
    conn = get_connection(db_path)
    try:
        if reset:
            drop_tables(conn)
        create_tables(conn)
        create_indexes(conn)
    finally:
        conn.close()


def insert_player_stats(
    conn: sqlite3.Connection,
    rows: Iterable[PlayerStatRow],
) -> int:
    """Insère les lignes de stats joueurs."""
    rows = list(rows)
    if not rows:
        return 0

    conn.executemany(
        """
        INSERT INTO players_stats (
            source_file, source_sheet, source_row_number,
            player, team_code, age, gp, wins, losses, minutes,
            pts, fgm, fga, fg_pct,
            three_pm, three_pa, three_p_pct,
            ftm, fta, ft_pct,
            oreb, dreb, reb, ast, tov, stl, blk, pf,
            fp, dd2, td3, plus_minus,
            offrtg, defrtg, netrtg,
            ast_pct, ast_to_ratio, ast_ratio,
            oreb_pct, dreb_pct, reb_pct,
            to_ratio, efg_pct, ts_pct, usg_pct, pace, pie, poss
        )
        VALUES (
            :source_file, :source_sheet, :source_row_number,
            :player, :team_code, :age, :gp, :wins, :losses, :minutes,
            :pts, :fgm, :fga, :fg_pct,
            :three_pm, :three_pa, :three_p_pct,
            :ftm, :fta, :ft_pct,
            :oreb, :dreb, :reb, :ast, :tov, :stl, :blk, :pf,
            :fp, :dd2, :td3, :plus_minus,
            :offrtg, :defrtg, :netrtg,
            :ast_pct, :ast_to_ratio, :ast_ratio,
            :oreb_pct, :dreb_pct, :reb_pct,
            :to_ratio, :efg_pct, :ts_pct, :usg_pct, :pace, :pie, :poss
        )
        """,
        [row.model_dump() for row in rows],
    )
    conn.commit()
    return len(rows)


def insert_teams(
    conn: sqlite3.Connection,
    rows: Iterable[TeamRow],
) -> int:
    """Insère les lignes de la table teams."""
    rows = list(rows)
    if not rows:
        return 0

    conn.executemany(
        """
        INSERT OR REPLACE INTO teams (
            team_code, team_name, source_file, source_sheet, source_row_number
        )
        VALUES (
            :team_code, :team_name, :source_file, :source_sheet, :source_row_number
        )
        """,
        [row.model_dump() for row in rows],
    )
    conn.commit()
    return len(rows)


def insert_metric_dictionary(
    conn: sqlite3.Connection,
    rows: Iterable[MetricDictionaryRow],
) -> int:
    """Insère les lignes du dictionnaire de métriques."""
    rows = list(rows)
    if not rows:
        return 0

    conn.executemany(
        """
        INSERT OR REPLACE INTO metric_dictionary (
            metric_code, metric_description, source_file, source_sheet, source_row_number
        )
        VALUES (
            :metric_code, :metric_description, :source_file, :source_sheet, :source_row_number
        )
        """,
        [row.model_dump() for row in rows],
    )
    conn.commit()
    return len(rows)


def count_rows(conn: sqlite3.Connection, table_name: str) -> int:
    """Retourne le nombre de lignes d'une table."""
    cursor = conn.execute(f"SELECT COUNT(*) AS n FROM {table_name}")
    row = cursor.fetchone()
    return int(row["n"])


def quick_summary(db_path: Path = DB_PATH) -> dict[str, int]:
    """Retourne un résumé rapide des volumes en base."""
    conn = get_connection(db_path)
    try:
        return {
            "players_stats": count_rows(conn, "players_stats"),
            "teams": count_rows(conn, "teams"),
            "metric_dictionary": count_rows(conn, "metric_dictionary"),
        }
    finally:
        conn.close()
