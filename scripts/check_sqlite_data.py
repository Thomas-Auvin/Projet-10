# scripts/check_sqlite_data.py
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.sql_db import DB_PATH, get_connection, quick_summary


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def fetch_all_dicts(conn, query: str):
    rows = conn.execute(query).fetchall()
    return [dict(r) for r in rows]


def main() -> int:
    if not DB_PATH.exists():
        print(f"Base introuvable : {DB_PATH}")
        print("Lance d'abord : uv run python scripts/ingest_excel_to_sqlite.py")
        return 1

    conn = get_connection(DB_PATH)
    try:
        print_section("Résumé global")
        print(quick_summary(DB_PATH))

        print_section("Aperçu players_stats (5 lignes)")
        rows = fetch_all_dicts(
            conn,
            """
            SELECT player, team_code, age, pts, three_p_pct, ts_pct
            FROM players_stats
            LIMIT 5
            """,
        )
        for row in rows:
            print(row)

        print_section("Top 5 scoreurs (PTS)")
        rows = fetch_all_dicts(
            conn,
            """
            SELECT player, team_code, pts
            FROM players_stats
            WHERE pts IS NOT NULL
            ORDER BY pts DESC
            LIMIT 5
            """,
        )
        for row in rows:
            print(row)

        print_section("Top 5 au pourcentage à 3 points (3P%)")
        rows = fetch_all_dicts(
            conn,
            """
            SELECT player, team_code, three_p_pct, three_pa
            FROM players_stats
            WHERE three_p_pct IS NOT NULL
            ORDER BY three_p_pct DESC
            LIMIT 5
            """,
        )
        for row in rows:
            print(row)

        print_section("Joueur le plus âgé / le plus jeune")
        oldest = fetch_all_dicts(
            conn,
            """
            SELECT player, team_code, age
            FROM players_stats
            WHERE age IS NOT NULL
            ORDER BY age DESC
            LIMIT 1
            """,
        )
        youngest = fetch_all_dicts(
            conn,
            """
            SELECT player, team_code, age
            FROM players_stats
            WHERE age IS NOT NULL
            ORDER BY age ASC
            LIMIT 1
            """,
        )
        print("Plus âgé :", oldest[0] if oldest else None)
        print("Plus jeune :", youngest[0] if youngest else None)

        print_section("Aperçu teams")
        rows = fetch_all_dicts(
            conn,
            """
            SELECT team_code, team_name
            FROM teams
            ORDER BY team_code
            LIMIT 10
            """,
        )
        for row in rows:
            print(row)

        print_section("Aperçu metric_dictionary")
        rows = fetch_all_dicts(
            conn,
            """
            SELECT metric_code, metric_description
            FROM metric_dictionary
            LIMIT 10
            """,
        )
        for row in rows:
            print(row)

        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
