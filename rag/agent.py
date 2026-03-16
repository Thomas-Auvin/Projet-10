# rag/agent.py
from __future__ import annotations

import asyncio
import logging
import threading
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.mistral import MistralModel

from rag.schemas import FinalAnswer, RetrievedContext, UserQuestion
from rag.sql_tool import ask_sql
from utils.config import MODEL_NAME, SEARCH_K
from utils.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


@dataclass
class AgentDeps:
    """Dépendances runtime injectées dans l'agent."""

    vector_store_manager: VectorStoreManager
    default_top_k: int = SEARCH_K

    # ---- Trace runtime pour l'évaluation ----
    used_tools: list[str] = field(default_factory=list)
    retrieved_source_ids: list[str] = field(default_factory=list)
    retrieved_contexts: list[str] = field(default_factory=list)
    sql_query: str | None = None
    sql_row_count: int | None = None
    sql_rows_preview: list[dict[str, Any]] = field(default_factory=list)
    sql_notes: str | None = None


def build_retrieved_context(
    *,
    question: str,
    vector_store_manager: VectorStoreManager,
    top_k: int,
) -> RetrievedContext:
    """Construit un contexte structuré à partir d'une question."""
    logger.info("build_retrieved_context(question=%r, top_k=%s)", question, top_k)

    chunks = vector_store_manager.search(question, k=top_k)

    if chunks:
        context_str = "\n\n---\n\n".join(
            [
                f"Source: {chunk.source} (Score: {chunk.score:.1f}%)\nContenu: {chunk.text}"
                for chunk in chunks
            ]
        )
    else:
        context_str = (
            "Aucune information pertinente trouvée dans la base de connaissances "
            "pour cette question."
        )

    return RetrievedContext(
        question=question,
        chunks=chunks,
        k=top_k,
        context_str=context_str,
    )


def build_agent() -> Agent[AgentDeps, FinalAnswer]:
    """
    Construit un agent neuf.

    Toute l'exécution réelle se fera sur une boucle asyncio persistante
    dans un thread dédié.
    """
    agent = Agent(
        MistralModel(MODEL_NAME),
        deps_type=AgentDeps,
        output_type=FinalAnswer,
        instructions=(
            "Tu es 'NBA Analyst AI', un assistant expert sur la NBA.\n"
            "\n"
            "Tu disposes de deux outils :\n"
            "1. retrieve_context : pour les questions textuelles, documentaires, Reddit, "
            "définitions, contexte narratif, avis exprimés dans les documents.\n"
            "2. sql_query_tool : pour les questions numériques, statistiques, top-k, "
            "comparaisons, moyennes, min/max, filtres, noms complets d'équipes à partir "
            "d'un code équipe, et définitions stockées dans les tables SQL.\n"
            "\n"
            "Règles de routage :\n"
            "- Si la question porte sur les documents Reddit, du texte libre, des opinions, "
            "un débat, ou un contexte narratif, utilise retrieve_context.\n"
            "- Si la question porte sur des statistiques joueurs/équipes, des classements, "
            "des extrêmes, des filtres, des agrégations, ou un code/nom d'équipe (ex: OKC), "
            "utilise sql_query_tool.\n"
            "- Si la question est hybride (par exemple définition + statistique, ou contexte "
            "textuel + valeur chiffrée), tu peux utiliser les deux outils.\n"
            "\n"
            "Consignes de réponse :\n"
            "- Appuie-toi sur les outils, pas sur ta mémoire.\n"
            "- Si le contexte ou les résultats sont insuffisants, signale-le clairement.\n"
            "- Le champ 'answer' doit être une chaîne de texte claire et concise.\n"
            "- Le champ 'used_retrieval' doit être True si au moins un outil a été utilisé "
            "pour construire la réponse, sinon False.\n"
            "- Le champ 'notes' peut servir à signaler une limite, un faible volume "
            "statistique, une ambiguïté, ou une contrainte métier.\n"
            "\n"
            "RÈGLE TRÈS IMPORTANTE SUR 'sources' :\n"
            "- 'sources' doit être une LISTE DE CHAÎNES DE CARACTÈRES UNIQUEMENT.\n"
            "- Ne mets jamais d'objet JSON, jamais de dictionnaire, jamais de clé/valeur.\n"
            "- Chaque élément de 'sources' doit être un simple texte.\n"
            "- Pour une réponse RAG, mets des chaînes comme les noms/fichiers/sources des chunks utilisés.\n"
            "- Pour une réponse SQL, mets uniquement des chaînes simples comme "
            "['SQL: players_stats'] ou ['SQL: teams', 'SQL: players_stats'].\n"
            "- Pour une réponse hybride, mélange uniquement des chaînes simples, par exemple "
            "['reddit_1.txt', 'SQL: players_stats'].\n"
            "- Format interdit : [{'name': 'sql_query_tool', 'query': '...'}].\n"
            "\n"
            "Exemples valides de sortie pour 'sources' :\n"
            "- ['reddit_1.txt']\n"
            "- ['SQL: players_stats']\n"
            "- ['reddit_3.txt', 'SQL: teams']\n"
            "\n"
            "Ta sortie finale doit respecter exactement le schéma FinalAnswer."
        ),
    )

    @agent.tool
    def retrieve_context(
        ctx: RunContext[AgentDeps],
        question: str,
        top_k: int | None = None,
    ) -> RetrievedContext:
        """Récupère les chunks les plus pertinents pour une question NBA."""
        k = top_k or ctx.deps.default_top_k
        logger.info("Tool retrieve_context appelé avec question='%s' k=%s", question, k)

        retrieved_context = build_retrieved_context(
            question=question,
            vector_store_manager=ctx.deps.vector_store_manager,
            top_k=k,
        )

        # ---- trace ----
        if "retrieve_context" not in ctx.deps.used_tools:
            ctx.deps.used_tools.append("retrieve_context")

        ctx.deps.retrieved_source_ids = [
            chunk.source
            for chunk in retrieved_context.chunks
            if getattr(chunk, "source", None)
        ]
        ctx.deps.retrieved_contexts = [
            chunk.text
            for chunk in retrieved_context.chunks
            if getattr(chunk, "text", None)
        ]

        logger.info(
            "Tool retrieve_context terminé: %s chunk(s) récupéré(s)",
            len(retrieved_context.chunks),
        )
        return retrieved_context

    @agent.tool
    def sql_query_tool(
        ctx: RunContext[AgentDeps],
        question: str,
    ) -> str:
        """
        Exécute une requête SQL dérivée de la question utilisateur et retourne
        un résumé texte compact.
        """
        logger.info("Tool sql_query_tool appelé avec question='%s'", question)

        result = ask_sql(question)
        preview_rows = result.rows[:10]

        # ---- trace ----
        if "sql_query_tool" not in ctx.deps.used_tools:
            ctx.deps.used_tools.append("sql_query_tool")

        ctx.deps.sql_query = result.sql_query
        ctx.deps.sql_row_count = result.row_count
        ctx.deps.sql_rows_preview = preview_rows
        ctx.deps.sql_notes = result.notes

        summary = (
            f"SQL_QUERY:\n{result.sql_query}\n\n"
            f"ROW_COUNT: {result.row_count}\n"
            f"ROWS_PREVIEW:\n{preview_rows}\n"
            f"NOTES: {result.notes or 'Aucune'}"
        )

        logger.info(
            "Tool sql_query_tool terminé: %s ligne(s) retournée(s)",
            result.row_count,
        )
        return summary

    return agent


async def run_agent_async(
    user_question: UserQuestion,
    vector_store_manager: VectorStoreManager,
    default_top_k: int = SEARCH_K,
) -> tuple[FinalAnswer, dict[str, Any]]:
    """Exécution async réelle de l'agent + trace."""
    effective_top_k = user_question.top_k or default_top_k

    deps = AgentDeps(
        vector_store_manager=vector_store_manager,
        default_top_k=effective_top_k,
    )

    logger.info(
        "Exécution de l'agent pour la question='%s' top_k=%s",
        user_question.question,
        effective_top_k,
    )

    prompt = (
        f"Question utilisateur : {user_question.question}\n"
        "Choisis le bon outil selon la nature de la question, puis réponds au format FinalAnswer.\n"
        "IMPORTANT: le champ sources doit être une liste de chaînes uniquement, jamais un objet."
    )

    agent = build_agent()
    result = await agent.run(prompt, deps=deps)

    trace = {
        "used_tools": deps.used_tools.copy(),
        "retrieved_source_ids": deps.retrieved_source_ids.copy(),
        "retrieved_contexts": deps.retrieved_contexts.copy(),
        "sql_query": deps.sql_query,
        "sql_row_count": deps.sql_row_count,
        "sql_rows_preview": deps.sql_rows_preview.copy(),
        "sql_notes": deps.sql_notes,
    }

    logger.info("Agent terminé avec succès.")
    return result.output, trace


class _PersistentAgentLoop:
    """
    Boucle asyncio persistante dans un thread dédié.

    Plus robuste que asyncio.run(...) à chaque question.
    """

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="p10-agent-loop",
            daemon=True,
        )
        self._thread.start()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def run(self, coro):
        future: Future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def stop(self) -> None:
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread.is_alive():
            self._thread.join(timeout=1)


_AGENT_LOOP = _PersistentAgentLoop()


def run_agent(
    user_question: UserQuestion,
    vector_store_manager: VectorStoreManager,
    default_top_k: int = SEARCH_K,
) -> FinalAnswer:
    """
    Point d'entrée synchrone pour Streamlit : renvoie seulement FinalAnswer.
    """
    final_answer, _trace = _AGENT_LOOP.run(
        run_agent_async(
            user_question=user_question,
            vector_store_manager=vector_store_manager,
            default_top_k=default_top_k,
        )
    )
    return final_answer


def run_agent_with_trace(
    user_question: UserQuestion,
    vector_store_manager: VectorStoreManager,
    default_top_k: int = SEARCH_K,
) -> tuple[FinalAnswer, dict[str, Any]]:
    """
    Variante pour l'évaluation : renvoie la réponse finale + la trace runtime.
    """
    return _AGENT_LOOP.run(
        run_agent_async(
            user_question=user_question,
            vector_store_manager=vector_store_manager,
            default_top_k=default_top_k,
        )
    )
