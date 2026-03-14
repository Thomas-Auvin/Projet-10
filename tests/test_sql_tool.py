# tests/test_sql_tool.py
from __future__ import annotations

from rag.sql_tool import ask_sql, is_safe_read_only_query


def test_is_safe_read_only_query_accepts_select():
    assert is_safe_read_only_query("SELECT * FROM players_stats LIMIT 5")


def test_is_safe_read_only_query_rejects_update():
    assert not is_safe_read_only_query("UPDATE players_stats SET pts = 0")


def test_is_safe_read_only_query_rejects_drop():
    assert not is_safe_read_only_query("DROP TABLE players_stats")


def test_ask_sql_top_scorer_integration():
    """
    Test d'intégration :
    - génère une requête SQL à partir d'une question
    - exécute la requête
    - vérifie qu'on récupère au moins une ligne
    """
    result = ask_sql("Qui a marqué le plus de points sur la saison ?")

    assert result.sql_query.lower().startswith("select")
    assert result.row_count >= 1
    assert isinstance(result.rows, list)
    assert "player" in result.rows[0] or "pts" in result.rows[0]


def test_ask_sql_oldest_player_integration():
    result = ask_sql("Quel est le joueur le plus âgé ?")

    assert result.sql_query.lower().startswith("select")
    assert result.row_count >= 1
    assert isinstance(result.rows, list)


def test_ask_sql_best_three_point_pct_integration():
    result = ask_sql("Qui a le meilleur pourcentage à 3 points ?")

    assert result.sql_query.lower().startswith("select")
    assert result.row_count >= 1
    assert isinstance(result.rows, list)
