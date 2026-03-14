from __future__ import annotations

import logging
import re
import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Any

from mistralai import Mistral

from rag.sql_db import DB_PATH
from rag.sql_schemas import SQLToolResult
from utils.config import MISTRAL_API_KEY, MODEL_NAME

logger = logging.getLogger(__name__)

FORBIDDEN_SQL_KEYWORDS = {
    "insert",
    "update",
    "delete",
    "drop",
    "alter",
    "create",
    "replace",
    "truncate",
    "attach",
    "detach",
    "pragma",
    "vacuum",
}

OUT_OF_SCOPE_PATTERNS = [
    r"\bdernier match\b",
    r"\b5 derniers matchs\b",
    r"\bderniers matchs\b",
    r"\bdomicile\b",
    r"\bextérieur\b",
    r"\bhome\b",
    r"\baway\b",
    r"\bvidéo\b",
    r"\bvideo\b",
    r"\blien\b",
    r"\bdunk\b",
    r"\bsemaine\b",
    r"\btransfert\b",
    r"\btransferts\b",
    r"\bchangé d['’]équipe\b",
    r"\bscore du dernier match\b",
]

JOIN_AMBIGUOUS_COLUMNS = {
    "team_code": "p.team_code",
    "player": "p.player",
    "pts": "p.pts",
    "ts_pct": "p.ts_pct",
    "netrtg": "p.netrtg",
    "offrtg": "p.offrtg",
    "defrtg": "p.defrtg",
    "minutes": "p.minutes",
    "gp": "p.gp",
    "wins": "p.wins",
    "losses": "p.losses",
    "three_pa": "p.three_pa",
    "three_pm": "p.three_pm",
    "three_p_pct": "p.three_p_pct",
    "reb": "p.reb",
    "ast": "p.ast",
    "stl": "p.stl",
    "blk": "p.blk",
    "tov": "p.tov",
    "plus_minus": "p.plus_minus",
    "usg_pct": "p.usg_pct",
    "pace": "p.pace",
    "pie": "p.pie",
    "ast_to_ratio": "p.ast_to_ratio",
}

SCORER_WORDS = (
    "meilleur marqueur",
    "meilleur scoreur",
    "top scoreur",
    "top marqueur",
    "best scorer",
    "leading scorer",
)


def detect_out_of_scope(question: str) -> str | None:
    q = question.lower().strip()
    for pattern in OUT_OF_SCOPE_PATTERNS:
        if re.search(pattern, q):
            return (
                "Question hors périmètre de la base SQLite actuelle. "
                "La base contient des statistiques joueurs/équipes et un dictionnaire de métriques, "
                "mais pas l’historique match par match, pas domicile/extérieur, pas vidéos, "
                "pas transferts ni actualité hebdomadaire."
            )
    return None


def normalize_question(question: str) -> str:
    return re.sub(r"\s+", " ", question.lower().strip())


def escape_sql_literal(value: str) -> str:
    return value.replace("'", "''")


def is_sql_joining_teams(sql_query: str) -> bool:
    sql_low = re.sub(r"\s+", " ", sql_query.lower())
    return " join teams " in sql_low


def extract_sql_from_text(text: str) -> str:
    """
    Extrait une requête SQL d'un texte LLM, même si le modèle renvoie :
    - un bloc ```sql ... ```
    - un bloc ``` ... ```
    - un bloc mal fermé
    - du texte avant le SELECT / WITH / OUT_OF_SCOPE
    """
    text = (text or "").strip()
    if not text:
        return ""

    code_block_match = re.search(r"```sql\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if code_block_match:
        text = code_block_match.group(1).strip()
    else:
        generic_code_block_match = re.search(r"```\s*(.*?)```", text, flags=re.DOTALL)
        if generic_code_block_match:
            text = generic_code_block_match.group(1).strip()
        else:
            # bloc ouvert sans fermeture
            text = re.sub(r"^\s*```sql\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"^\s*```\s*", "", text)
            text = re.sub(r"\s*```\s*$", "", text)

    text = text.strip().strip("`").strip()

    # garde uniquement à partir du premier mot utile
    anchor = re.search(r"\b(OUT_OF_SCOPE|SELECT|WITH)\b", text, flags=re.IGNORECASE)
    if anchor:
        text = text[anchor.start():].strip()

    # supprime un éventuel préfixe "sql"
    text = re.sub(r"^\s*sql\s*", "", text, flags=re.IGNORECASE).strip()

    # supprime les ; finaux parasites
    text = re.sub(r";+\s*$", "", text).strip()

    return text


def normalize_generated_sql(sql_query: str) -> str:
    return extract_sql_from_text(sql_query)


def qualify_ambiguous_columns(sql_query: str) -> str:
    """
    Correction légère si le LLM a joint 'teams' mais a laissé des colonnes non qualifiées.
    On ne touche que les colonnes connues et seulement si la requête utilise un alias p/t.
    """
    if not is_sql_joining_teams(sql_query):
        return sql_query

    normalized = re.sub(r"\s+", " ", sql_query.strip()).lower()
    if " players_stats p " not in f" {normalized} ":
        return sql_query

    fixed = sql_query
    for raw_col, qualified_col in JOIN_AMBIGUOUS_COLUMNS.items():
        fixed = re.sub(rf"(?<!\.)\b{raw_col}\b", qualified_col, fixed)

    fixed = re.sub(r"(?<!\.)\bteam_name\b", "t.team_name", fixed)
    return fixed


def _normalize_dangling_table_aliases(sql_query: str) -> str:
    """
    Supprime les alias p. / t. quand ils sont utilisés sans alias déclaré
    dans le FROM/JOIN principal.

    Exemples :
    - FROM players_stats + WHERE p.gp > 10  -> WHERE gp > 10
    - JOIN teams + WHERE t.team_code='OKC' -> WHERE team_code='OKC'

    On ne touche pas aux alias si `players_stats p` ou `teams t` existent bien.
    """
    sql = sql_query

    has_players_alias = bool(
        re.search(r"\bFROM\s+players_stats\s+(?:AS\s+)?p\b", sql, flags=re.IGNORECASE)
    ) or bool(
        re.search(r"\bJOIN\s+players_stats\s+(?:AS\s+)?p\b", sql, flags=re.IGNORECASE)
    )

    has_teams_alias = bool(
        re.search(r"\bFROM\s+teams\s+(?:AS\s+)?t\b", sql, flags=re.IGNORECASE)
    ) or bool(
        re.search(r"\bJOIN\s+teams\s+(?:AS\s+)?t\b", sql, flags=re.IGNORECASE)
    )

    if not has_players_alias:
        sql = re.sub(r"\bp\.", "", sql)

    if not has_teams_alias:
        sql = re.sub(r"\bt\.", "", sql)

    return sql


def _question_expects_single_row(question: str) -> bool:
    """
    Détecte les questions qui demandent UNE seule réponse / un seul joueur /
    une seule valeur extrême.
    """
    q = question.lower()

    single_patterns = [
        r"\bquel est le joueur\b",
        r"\bqui a le meilleur\b",
        r"\bqui a la meilleure\b",
        r"\bqui a le plus\b",
        r"\bqui a le moins\b",
        r"\bquel joueur a\b",
        r"\bquel joueur est\b",
        r"\ble meilleur\b",
        r"\bla meilleure\b",
        r"\ble plus élevé\b",
        r"\bla plus élevée\b",
        r"\ble plus haut\b",
        r"\bla plus haute\b",
        r"\ble plus faible\b",
        r"\bla plus faible\b",
        r"\ble plus jeune\b",
        r"\ble plus âgé\b",
        r"\bquel curry\b",
        r"\bquel est le nom complet\b",
    ]

    return any(re.search(pattern, q) for pattern in single_patterns)


def _has_top_level_limit(sql_query: str) -> bool:
    """
    Détecte un LIMIT au niveau principal de la requête.
    Ignore les LIMIT présents dans des sous-requêtes.
    """
    depth = 0
    upper_sql = sql_query.upper()

    i = 0
    while i < len(upper_sql):
        ch = upper_sql[i]

        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif depth == 0 and upper_sql.startswith("LIMIT", i):
            before_ok = i == 0 or not upper_sql[i - 1].isalnum()
            after_idx = i + len("LIMIT")
            after_ok = after_idx >= len(upper_sql) or not upper_sql[after_idx].isalnum()
            if before_ok and after_ok:
                return True
        i += 1

    return False


def _is_single_row_aggregate(sql_query: str) -> bool:
    """
    Heuristique simple :
    si la requête principale contient un agrégat et pas de GROUP BY,
    on considère qu'elle renvoie une seule ligne.
    """
    sql = sql_query.upper()

    has_aggregate = any(func in sql for func in ["MAX(", "MIN(", "AVG(", "SUM(", "COUNT("])
    has_group_by = "GROUP BY" in sql

    return has_aggregate and not has_group_by


def _enforce_single_row_shape(question: str, sql_query: str) -> str:
    """
    Si la question attend une seule ligne et que la requête n'impose pas déjà
    cette forme (LIMIT top-level ou agrégat simple), ajoute LIMIT 1.
    """
    if not _question_expects_single_row(question):
        return sql_query

    if _has_top_level_limit(sql_query):
        return sql_query

    if _is_single_row_aggregate(sql_query):
        return sql_query

    sql = sql_query.rstrip().rstrip(";")
    return f"{sql}\nLIMIT 1"


def _postprocess_generated_sql(question: str, sql_query: str) -> str:
    """
    Post-traitement local, déterministe et peu risqué :
    1) supprime les alias pendants p./t.
    2) force une forme mono-ligne pour les questions de type superlatif / unique
    """
    sql = sql_query.strip()

    # Retire d'éventuelles fences markdown
    sql = re.sub(r"^```sql\s*", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"^```\s*", "", sql)
    sql = re.sub(r"\s*```$", "", sql)

    sql = _normalize_dangling_table_aliases(sql)
    sql = _enforce_single_row_shape(question, sql)

    return sql


@lru_cache(maxsize=8)
def get_team_catalog(db_path_str: str) -> list[tuple[str, str]]:
    conn = sqlite3.connect(db_path_str)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT team_code, team_name FROM teams ORDER BY team_name"
        ).fetchall()
        return [(str(r["team_code"]), str(r["team_name"])) for r in rows]
    finally:
        conn.close()


def find_team_reference(question: str, db_path: Path = DB_PATH) -> tuple[str, str] | None:
    q = normalize_question(question)
    catalog = get_team_catalog(str(db_path))

    # priorité au nom complet
    for team_code, team_name in catalog:
        if team_name.lower() in q:
            return team_code, team_name

    # puis code équipe
    for team_code, team_name in catalog:
        if re.search(rf"\b{re.escape(team_code.lower())}\b", q):
            return team_code, team_name

    return None


def is_scorer_question(question: str) -> bool:
    q = normalize_question(question)
    return any(word in q for word in SCORER_WORDS)


def is_points_per_minute_question(question: str) -> bool:
    q = normalize_question(question)
    return (
        "par minute" in q
        or "pts / (gp×min)" in q
        or "pts / (gp×min)" in q.replace(" ", "")
        or "pts / (gp*min)" in q
        or "points par minute" in q
    )


def is_top10_points_average_question(question: str) -> bool:
    q = normalize_question(question)
    return (
        ("moyenne" in q or "average" in q)
        and ("top 10" in q or "10 meilleurs" in q or "10 meilleur" in q)
        and ("points" in q or "pts" in q)
        and ("marqueur" in q or "scoreur" in q or "pts" in q)
    )


def is_top1_vs_top10_gap_question(question: str) -> bool:
    q = normalize_question(question)
    return (
        ("écart" in q or "ecart" in q or "difference" in q)
        and ("10e" in q or "10ème" in q or "10eme" in q or "#10" in q)
        and ("meilleur marqueur" in q or "meilleur scoreur" in q or "pts" in q or "points" in q)
    )


def build_rule_based_sql(question: str, db_path: Path = DB_PATH) -> str | None:
    """
    Quelques overrides déterministes pour les cas les plus coûteux en score :
    - moyenne top 10 PTS
    - écart #1 / #10 PTS
    - meilleur marqueur / scoreur (global ou d'une équipe)
    """
    if is_top10_points_average_question(question):
        return """
SELECT AVG(pts) AS avg_pts_top_10
FROM (
    SELECT pts
    FROM players_stats
    ORDER BY pts DESC
    LIMIT 10
)
""".strip()

    if is_top1_vs_top10_gap_question(question):
        return """
SELECT
    (SELECT pts FROM players_stats ORDER BY pts DESC LIMIT 1)
    - (SELECT pts FROM players_stats ORDER BY pts DESC LIMIT 1 OFFSET 9)
    AS pts_gap
""".strip()

    if is_scorer_question(question) and not is_points_per_minute_question(question):
        team_ref = find_team_reference(question, db_path=db_path)
        if team_ref is not None:
            team_code, _team_name = team_ref
            return f"""
SELECT p.player, p.pts
FROM players_stats p
JOIN teams t ON p.team_code = t.team_code
WHERE t.team_code = '{escape_sql_literal(team_code)}'
ORDER BY p.pts DESC
LIMIT 1
""".strip()

        return """
SELECT player, pts
FROM players_stats
ORDER BY pts DESC
LIMIT 1
""".strip()

    return None


def validate_sql_semantics(question: str, sql_query: str) -> str | None:
    q = normalize_question(question)
    sql_low = re.sub(r"\s+", " ", sql_query.lower())

    # GP × Min >= 1000 doit rester une contrainte sur le produit, pas uniquement minutes >= 1000
    if "gp" in q and "min" in q and "1000" in q:
        acceptable_patterns = [
            "gp * minutes >= 1000",
            "(gp * minutes) >= 1000",
            "p.gp * p.minutes >= 1000",
            "(p.gp * p.minutes) >= 1000",
        ]
        if not any(pat in sql_low for pat in acceptable_patterns):
            return (
                "La question impose une contrainte sur les minutes totales GP×Min ≥ 1000, "
                "mais le SQL généré ne respecte pas cette formule."
            )

    # Sur jointure teams, les colonnes communes doivent être qualifiées
    if "join teams" in sql_low:
        for col in ["team_code", "player", "pts", "ts_pct", "netrtg", "offrtg", "defrtg"]:
            if re.search(rf"(?<!\.)\b{col}\b", sql_low):
                return (
                    f"Requête potentiellement ambiguë après JOIN teams : colonne '{col}' non qualifiée. "
                    "Utiliser des alias explicites p. / t."
                )

    # Règle métier : meilleur marqueur / scoreur = pts, pas pts par minute
    if is_scorer_question(question) and not is_points_per_minute_question(question):
        if "pts_per_minute" in sql_low or re.search(r"pts\s*/\s*\(", sql_low):
            return (
                "La question demande un meilleur marqueur / scoreur, donc il faut classer sur pts "
                "et non sur des points par minute."
            )
        if " order by " in sql_low and "pts desc" not in sql_low and "p.pts desc" not in sql_low:
            return (
                "La question demande un meilleur marqueur / scoreur, il faut donc ordonner par pts DESC."
            )

    # Règle métier : moyenne top 10 points
    if is_top10_points_average_question(question):
        if "avg(" not in sql_low:
            return "La question demande une moyenne ; la requête doit utiliser AVG(...)."
        if "limit 10" not in sql_low:
            return "La question porte sur le top 10 ; la requête doit limiter explicitement à 10 lignes."
        if "from (" not in sql_low and "with " not in sql_low:
            return (
                "La moyenne doit porter sur le top 10, donc il faut une sous-requête ou un CTE."
            )

    return None


def get_sqlite_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Ouvre une connexion SQLite en lecture."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def get_schema_description() -> str:
    """Description textuelle minimale du schéma SQL exposé au LLM."""
    return """
Base SQLite disponible : nba.sqlite

Table players_stats
- player (TEXT)
- team_code (TEXT)
- age (INTEGER)
- gp (INTEGER)
- wins (INTEGER)
- losses (INTEGER)
- minutes (REAL)
- pts (REAL)
- fgm (REAL)
- fga (REAL)
- fg_pct (REAL)
- three_pm (REAL)
- three_pa (REAL)
- three_p_pct (REAL)
- ftm (REAL)
- fta (REAL)
- ft_pct (REAL)
- oreb (REAL)
- dreb (REAL)
- reb (REAL)
- ast (REAL)
- tov (REAL)
- stl (REAL)
- blk (REAL)
- pf (REAL)
- fp (REAL)
- dd2 (REAL)
- td3 (REAL)
- plus_minus (REAL)
- offrtg (REAL)
- defrtg (REAL)
- netrtg (REAL)
- ast_pct (REAL)
- ast_to_ratio (REAL)
- ast_ratio (REAL)
- oreb_pct (REAL)
- dreb_pct (REAL)
- reb_pct (REAL)
- to_ratio (REAL)
- efg_pct (REAL)
- ts_pct (REAL)
- usg_pct (REAL)
- pace (REAL)
- pie (REAL)
- poss (REAL)

Table teams
- team_code (TEXT)
- team_name (TEXT)

Table metric_dictionary
- metric_code (TEXT)
- metric_description (TEXT)
""".strip()


def build_sql_system_prompt() -> str:
    return f"""
Tu es un assistant expert SQL.
Ta tâche est de transformer une question utilisateur en UNE requête SQL SQLite valide.

Contraintes absolues :
- retourne uniquement UNE requête SQL
- uniquement du SELECT
- aucun commentaire
- aucun texte explicatif
- aucun markdown, aucun bloc ```sql
- utilise uniquement les tables et colonnes du schéma fourni
- si la question ne peut pas être traitée avec la base, retourne exactement :
OUT_OF_SCOPE

Schéma :
{get_schema_description()}

Règles obligatoires :
1. Produis UNE SEULE requête SQL SQLite.
2. Requête STRICTEMENT en lecture seule : SELECT uniquement.
3. Si la question ne peut pas être répondue avec ces tables, réponds exactement : OUT_OF_SCOPE
4. Si tu fais une jointure avec teams, utilise obligatoirement :
   FROM players_stats p
   JOIN teams t ON p.team_code = t.team_code
5. En cas de jointure, toutes les colonnes de players_stats doivent être préfixées par p.
6. team_name doit être préfixé par t.
7. N’invente jamais de colonnes.
8. Pour les contraintes du type GP×Min ≥ 1000, traduire exactement par :
   p.gp * p.minutes >= 1000
9. N’utilise JOIN teams que si nécessaire :
   - récupérer team_name
   - filtrer sur team_name
10. Si la question demande :
   - derniers matchs
   - domicile / extérieur
   - vidéos / liens
   - transferts
   - score d’un match
   alors réponds OUT_OF_SCOPE.
11. Si la question parle de "top 10", "parmi les 10 meilleurs", ou d’un sous-ensemble ordonné,
   respecte explicitement cette logique avec une sous-requête ou un CTE si nécessaire.
12. Si la question demande un calcul par minute avec GP×Min, utilise bien gp * minutes
   et non minutes seul.
13. Si la question demande le "meilleur marqueur" ou "meilleur scoreur", cela signifie :
   ORDER BY pts DESC
   sauf si la question mentionne explicitement "par minute".
14. Si tu écris une sous-requête, n'utilise pas d'alias de table externe invalide
   dans la requête englobante.
   Exemple correct :
   SELECT AVG(pts) FROM (SELECT pts FROM players_stats ORDER BY pts DESC LIMIT 10)
15. Ne renvoie jamais ```sql ni ``` ni du texte autour.
""".strip()


def is_safe_read_only_query(sql_query: str) -> bool:
    """Vérifie qu'une requête est strictement en lecture seule."""
    if not sql_query:
        return False

    normalized = re.sub(r"\s+", " ", sql_query.strip().lower())

    if not (normalized.startswith("select") or normalized.startswith("with")):
        return False

    for keyword in FORBIDDEN_SQL_KEYWORDS:
        if re.search(rf"\b{re.escape(keyword)}\b", normalized):
            return False

    return True


def run_sql_query(sql_query: str, db_path: Path = DB_PATH) -> list[dict[str, Any]]:
    """Exécute une requête SQL en lecture seule et retourne les lignes."""
    if not is_safe_read_only_query(sql_query):
        raise ValueError("Requête SQL refusée : seules les requêtes SELECT / WITH sont autorisées.")

    conn = get_sqlite_connection(db_path)
    try:
        rows = conn.execute(sql_query).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def _call_mistral_for_sql(user_prompt: str) -> str:
    if not MISTRAL_API_KEY:
        raise RuntimeError("MISTRAL_API_KEY manquante pour la génération SQL.")

    client = Mistral(api_key=MISTRAL_API_KEY)

    response = client.chat.complete(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": build_sql_system_prompt()},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
    )

    if not response.choices:
        raise RuntimeError("Aucune réponse SQL reçue du modèle.")

    raw_text = response.choices[0].message.content or ""
    return normalize_generated_sql(raw_text).strip()


def generate_sql_query(question: str, db_path: Path = DB_PATH) -> str:
    """Génère une requête SQL à partir d'une question en langage naturel."""
    rule_based = build_rule_based_sql(question, db_path=db_path)
    if rule_based is not None:
        logger.info("SQL déterministe utilisé pour la question '%s': %s", question, rule_based)
        return rule_based

    sql_query = _call_mistral_for_sql(f"Question utilisateur : {question}")
    sql_query = normalize_generated_sql(sql_query)
    sql_query = _postprocess_generated_sql(question, sql_query)
    logger.info("SQL généré pour la question '%s': %s", question, sql_query)
    return sql_query


def repair_sql_query(
    question: str,
    previous_sql: str,
    error_message: str,
    db_path: Path = DB_PATH,
) -> str:
    """
    Demande une correction ciblée au modèle après erreur SQL/validation.
    On essaie d'abord une réparation locale déterministe sur les cas connus.
    """
    rule_based = build_rule_based_sql(question, db_path=db_path)
    if rule_based is not None:
        logger.info("SQL réparé localement pour la question '%s': %s", question, rule_based)
        return rule_based

    cleaned_previous = normalize_generated_sql(previous_sql)
    cleaned_previous = _postprocess_generated_sql(question, cleaned_previous)
    if cleaned_previous != previous_sql and is_safe_read_only_query(cleaned_previous):
        logger.info(
            "SQL nettoyé localement avant réparation LLM pour la question '%s': %s",
            question,
            cleaned_previous,
        )
        previous_sql = cleaned_previous

    repair_prompt = f"""
Question utilisateur : {question}

SQL précédent :
{previous_sql}

Erreur observée :
{error_message}

Corrige la requête SQL.

Rappels impératifs :
- retourne uniquement le SQL
- SELECT ou WITH uniquement
- aucun markdown, aucun ```sql
- si la question est hors périmètre, retourne exactement : OUT_OF_SCOPE
- si la question demande GP×Min ≥ 1000, conserve exactement cette contrainte
- si la question demande "meilleur marqueur" / "meilleur scoreur", ordonne par pts DESC
  sauf si la question dit explicitement "par minute"
- si tu fais une sous-requête, n'utilise pas un alias de table externe invalide
  dans la requête englobante
- en cas de JOIN teams, qualifie correctement p. et t.
""".strip()

    repaired = _call_mistral_for_sql(repair_prompt)
    repaired = normalize_generated_sql(repaired)
    repaired = _postprocess_generated_sql(question, repaired)
    logger.info("SQL réparé pour la question '%s': %s", question, repaired)
    return repaired


def _make_result(
    *,
    question: str,
    sql_query: str | None,
    rows: list[dict[str, Any]],
    notes: str | None,
) -> SQLToolResult:
    safe_sql_query = sql_query if isinstance(sql_query, str) and sql_query.strip() else "OUT_OF_SCOPE"

    return SQLToolResult(
        question=question,
        sql_query=safe_sql_query,
        rows=rows,
        row_count=len(rows),
        notes=notes,
    )


def ask_sql(question: str, db_path: Path = DB_PATH) -> SQLToolResult:
    """
    Pipeline principal NL -> SQL -> résultats.
    Retourne un objet structuré pour l'agent.
    """
    refusal_reason = detect_out_of_scope(question)
    if refusal_reason is not None:
        logger.info("Question refusée car hors périmètre SQL: %s", question)
        return _make_result(
            question=question,
            sql_query="OUT_OF_SCOPE",
            rows=[],
            notes=refusal_reason,
        )

    try:
        sql_query = generate_sql_query(question, db_path=db_path)
    except Exception as e:
        logger.exception("Erreur pendant la génération SQL pour la question '%s'", question)
        return _make_result(
            question=question,
            sql_query="OUT_OF_SCOPE",
            rows=[],
            notes=f"Erreur lors de la génération SQL : {e}",
        )

    sql_query = normalize_generated_sql(sql_query)
    sql_query = _postprocess_generated_sql(question, sql_query)

    if sql_query == "OUT_OF_SCOPE":
        return _make_result(
            question=question,
            sql_query="OUT_OF_SCOPE",
            rows=[],
            notes="Question hors périmètre de la base SQLite actuelle.",
        )

    if sql_query.lower().startswith("select 'unanswerable_with_sql'"):
        return _make_result(
            question=question,
            sql_query=sql_query,
            rows=[],
            notes="Question non traitable avec la base SQL actuelle.",
        )

    sql_query = qualify_ambiguous_columns(sql_query)

    semantic_error = validate_sql_semantics(question, sql_query)
    if semantic_error is not None:
        logger.warning("SQL rejeté avant exécution pour '%s' : %s", question, semantic_error)
        try:
            repaired_sql = repair_sql_query(question, sql_query, semantic_error, db_path=db_path)
        except Exception as e:
            logger.exception("Erreur pendant la réparation SQL pour '%s'", question)
            return _make_result(
                question=question,
                sql_query=sql_query,
                rows=[],
                notes=f"Erreur lors de la réparation SQL : {e}",
            )

        if repaired_sql == "OUT_OF_SCOPE":
            return _make_result(
                question=question,
                sql_query="OUT_OF_SCOPE",
                rows=[],
                notes="Question non faisable avec la base SQLite actuelle.",
            )

        sql_query = normalize_generated_sql(repaired_sql)
        sql_query = _postprocess_generated_sql(question, sql_query)
        sql_query = qualify_ambiguous_columns(sql_query)

        semantic_error = validate_sql_semantics(question, sql_query)
        if semantic_error is not None:
            return _make_result(
                question=question,
                sql_query=sql_query,
                rows=[],
                notes=f"SQL encore invalide sémantiquement après réparation : {semantic_error}",
            )

    if not is_safe_read_only_query(sql_query):
        return _make_result(
            question=question,
            sql_query=sql_query,
            rows=[],
            notes="La requête générée a été refusée car elle n'est pas strictement en lecture seule.",
        )

    try:
        rows = run_sql_query(sql_query, db_path=db_path)
        return _make_result(
            question=question,
            sql_query=sql_query,
            rows=rows,
            notes=None if rows else "La requête est valide mais n'a retourné aucune ligne.",
        )

    except sqlite3.OperationalError as e:
        logger.exception("Erreur SQL opérationnelle pour la question '%s'", question)

        try:
            repaired_sql = repair_sql_query(question, sql_query, str(e), db_path=db_path)
        except Exception as repair_exc:
            logger.exception("Erreur pendant la réparation SQL pour '%s'", question)
            return _make_result(
                question=question,
                sql_query=sql_query,
                rows=[],
                notes=f"Erreur lors de l'exécution SQL : {e}. Réparation impossible : {repair_exc}",
            )

        if repaired_sql == "OUT_OF_SCOPE":
            return _make_result(
                question=question,
                sql_query="OUT_OF_SCOPE",
                rows=[],
                notes="Question non faisable avec la base SQLite actuelle.",
            )

        repaired_sql = normalize_generated_sql(repaired_sql)
        repaired_sql = _postprocess_generated_sql(question, repaired_sql)
        repaired_sql = qualify_ambiguous_columns(repaired_sql)

        if not is_safe_read_only_query(repaired_sql):
            return _make_result(
                question=question,
                sql_query=repaired_sql,
                rows=[],
                notes="La requête SQL réparée a été refusée car elle n'est pas strictement en lecture seule.",
            )

        semantic_error = validate_sql_semantics(question, repaired_sql)
        if semantic_error is not None:
            return _make_result(
                question=question,
                sql_query=repaired_sql,
                rows=[],
                notes=f"SQL réparé encore invalide sémantiquement : {semantic_error}",
            )

        try:
            rows = run_sql_query(repaired_sql, db_path=db_path)
            return _make_result(
                question=question,
                sql_query=repaired_sql,
                rows=rows,
                notes=None if rows else "La requête SQL réparée est valide mais n'a retourné aucune ligne.",
            )
        except Exception as e2:
            logger.exception("Échec final SQL pour la question '%s'", question)
            return _make_result(
                question=question,
                sql_query=repaired_sql,
                rows=[],
                notes=f"Erreur lors de l'exécution SQL après réparation : {e2}",
            )

    except Exception as e:
        logger.exception("Erreur pendant l'exécution SQL pour la question '%s'", question)
        return _make_result(
            question=question,
            sql_query=sql_query,
            rows=[],
            notes=f"Erreur lors de l'exécution SQL : {e}",
        )
