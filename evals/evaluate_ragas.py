from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from dotenv import load_dotenv
from pydantic_ai.exceptions import ModelHTTPError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.config import MISTRAL_API_KEY, MODEL_NAME, SEARCH_K
from utils.vector_store import VectorStoreManager
from rag.agent import run_agent_with_trace
from rag.schemas import UserQuestion

from ragas import EvaluationDataset, evaluate

try:
    from ragas.run_config import RunConfig
except Exception:
    RunConfig = None  # type: ignore

from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings


# -----------------------------
# Safe LangChain Mistral wrapper
# -----------------------------
class SafeChatMistralAI(ChatMistralAI):
    """
    Patch défensif pour éviter les erreurs du type:
    TypeError: unsupported operand type(s) for +=: 'dict' and 'dict'

    On ignore les valeurs non numériques dans token_usage lors de l'agrégation.
    """

    def _combine_llm_outputs(self, llm_outputs: list[dict | None]) -> dict:
        overall_token_usage: dict[str, float] = {}

        for output in llm_outputs:
            if output is None:
                continue

            token_usage = output.get("token_usage")
            if not token_usage:
                continue

            for k, v in token_usage.items():
                if isinstance(v, (int, float)):
                    overall_token_usage[k] = overall_token_usage.get(k, 0) + v
                else:
                    continue

        return {
            "token_usage": overall_token_usage,
            "model_name": self.model,
        }


# -----------------------------
# Utilities
# -----------------------------
def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def normalize_source_id(s: str) -> str:
    s = (s or "").strip().replace("\\", "/")
    return Path(s).name


def retrieval_metrics(
    expected_sources: List[str], retrieved_sources_ranked: List[str]
) -> Dict[str, Any]:
    expected = [normalize_source_id(s) for s in (expected_sources or []) if s]
    if len(expected) == 0:
        return {
            "hit_at_k": None,
            "precision_at_k": None,
            "recall_at_k": None,
            "mrr": None,
        }

    retrieved = [normalize_source_id(s) for s in retrieved_sources_ranked if s]
    k = len(retrieved) if len(retrieved) > 0 else 0
    expected_set = set(expected)

    seen = set()
    retrieved_unique = []
    for s in retrieved:
        if s not in seen:
            seen.add(s)
            retrieved_unique.append(s)

    hits = [s for s in retrieved_unique if s in expected_set]
    hit_at_k = 1 if len(hits) > 0 else 0
    precision = (len(hits) / k) if k > 0 else 0.0
    recall = (len(hits) / len(expected_set)) if len(expected_set) > 0 else 0.0

    mrr = 0.0
    for i, s in enumerate(retrieved_unique, start=1):
        if s in expected_set:
            mrr = 1.0 / i
            break

    return {
        "hit_at_k": hit_at_k,
        "precision_at_k": precision,
        "recall_at_k": recall,
        "mrr": mrr,
    }


def normalize_tool_name(name: str) -> str:
    raw = (name or "").strip().lower()

    mapping = {
        "rag": "retrieve_context",
        "retrieval": "retrieve_context",
        "retrieve": "retrieve_context",
        "retrieve_context": "retrieve_context",
        "context": "retrieve_context",
        "text": "retrieve_context",
        "doc": "retrieve_context",
        "reddit": "retrieve_context",
        "sql": "sql_query_tool",
        "sql_tool": "sql_query_tool",
        "sql_query_tool": "sql_query_tool",
        "database": "sql_query_tool",
        "db": "sql_query_tool",
        "numeric": "sql_query_tool",
        "hybrid": "hybrid",
        "both": "hybrid",
    }

    return mapping.get(raw, raw)


def parse_expected_tool(expected_tool: Any) -> Set[str]:
    """
    Convertit expected_tool en ensemble canonique.
    Exemples:
    - "rag" -> {"retrieve_context"}
    - "sql" -> {"sql_query_tool"}
    - "hybrid" -> {"retrieve_context", "sql_query_tool"}
    - "rag+sql" -> {"retrieve_context", "sql_query_tool"}
    """
    if expected_tool is None:
        return set()

    if isinstance(expected_tool, list):
        parts = [str(x) for x in expected_tool]
    else:
        raw = str(expected_tool)
        for sep in [",", ";", "|", "+", "/"]:
            raw = raw.replace(sep, " ")
        parts = raw.split()

    normalized = {normalize_tool_name(p) for p in parts if str(p).strip()}

    if "hybrid" in normalized:
        return {"retrieve_context", "sql_query_tool"}

    final = set()
    for item in normalized:
        if item in {"retrieve_context", "sql_query_tool"}:
            final.add(item)

    return final


def tool_routing_metrics(expected_tool: Any, used_tools: List[str]) -> Dict[str, Any]:
    expected_set = parse_expected_tool(expected_tool)
    used_set = {normalize_tool_name(x) for x in (used_tools or [])}

    if len(expected_set) == 0:
        return {
            "expected_tool_set": [],
            "used_tool_set": sorted(used_set),
            "tool_routing_ok": None,
            "tool_call_accuracy_ragas": None,
            "tool_call_precision_ragas": None,
            "tool_call_recall_ragas": None,
            "tool_call_f1_ragas": None,
        }

    inter = expected_set & used_set
    ok = 1 if expected_set.issubset(used_set) else 0

    precision = len(inter) / len(used_set) if len(used_set) > 0 else 0.0
    recall = len(inter) / len(expected_set) if len(expected_set) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "expected_tool_set": sorted(expected_set),
        "used_tool_set": sorted(used_set),
        "tool_routing_ok": ok,
        "tool_call_accuracy_ragas": ok,
        "tool_call_precision_ragas": precision,
        "tool_call_recall_ragas": recall,
        "tool_call_f1_ragas": f1,
    }


def build_eval_contexts_from_trace(trace: Dict[str, Any]) -> List[str]:
    """
    Construit les 'retrieved_contexts' pour RAGAS à partir de:
    - chunks RAG
    - sortie SQL résumée

    Très important pour les questions numériques:
    on donne à faithfulness une "preuve" SQL au lieu de laisser les contexts vides.
    """
    contexts: List[str] = []

    rag_contexts = trace.get("retrieved_contexts") or []
    for c in rag_contexts:
        if c and str(c).strip():
            contexts.append(str(c))

    sql_query = trace.get("sql_query")
    sql_rows_preview = trace.get("sql_rows_preview") or []
    sql_row_count = trace.get("sql_row_count")
    sql_notes = trace.get("sql_notes")

    if sql_query or sql_rows_preview:
        sql_context = (
            f"SQL_QUERY: {sql_query or 'N/A'}\n"
            f"SQL_ROW_COUNT: {sql_row_count}\n"
            f"SQL_ROWS_PREVIEW: {sql_rows_preview}\n"
            f"SQL_NOTES: {sql_notes or 'Aucune'}"
        )
        contexts.append(sql_context)

    return contexts


def extract_answer_text(final_answer: Any) -> str:
    """
    Extrait le texte final depuis FinalAnswer.
    Fallback défensif selon le schéma réel.
    """
    for attr in ["answer", "response", "final_answer", "content"]:
        if hasattr(final_answer, attr):
            value = getattr(final_answer, attr)
            if value is not None:
                return str(value)

    return str(final_answer)


def extract_sources(final_answer: Any) -> List[str]:
    if hasattr(final_answer, "sources"):
        value = getattr(final_answer, "sources")
        if isinstance(value, list):
            return [str(x) for x in value]
    return []


def extract_notes(final_answer: Any) -> Any:
    if hasattr(final_answer, "notes"):
        return getattr(final_answer, "notes")
    return None


def extract_used_retrieval(final_answer: Any) -> Any:
    if hasattr(final_answer, "used_retrieval"):
        return getattr(final_answer, "used_retrieval")
    return None


# -----------------------------
# RAGAS metrics
# -----------------------------
def _import_metric_classes() -> Tuple[Any, Any]:
    """
    Force les métriques LEGACY de ragas.metrics
    pour rester compatibles avec LangchainLLMWrapper / LangchainEmbeddingsWrapper.
    """
    try:
        from ragas.metrics import Faithfulness, ResponseRelevancy

        return Faithfulness, ResponseRelevancy
    except Exception:
        pass

    try:
        from ragas.metrics import Faithfulness, AnswerRelevancy

        return Faithfulness, AnswerRelevancy
    except Exception:
        pass

    try:
        from ragas import metrics as ragas_metrics

        faith_obj = None
        rel_obj = None

        for name in ("Faithfulness", "faithfulness"):
            if hasattr(ragas_metrics, name):
                faith_obj = getattr(ragas_metrics, name)
                break

        for name in (
            "ResponseRelevancy",
            "AnswerRelevancy",
            "response_relevancy",
            "answer_relevancy",
        ):
            if hasattr(ragas_metrics, name):
                rel_obj = getattr(ragas_metrics, name)
                break

        if faith_obj is not None and rel_obj is not None:
            return faith_obj, rel_obj
    except Exception:
        pass

    try:
        from ragas.metrics.collections import Faithfulness, AnswerRelevancy

        return Faithfulness, AnswerRelevancy
    except Exception as e:
        raise ImportError(
            "Impossible de charger Faithfulness / Relevancy depuis RAGAS."
        ) from e


def build_core_ragas_metrics(
    eval_llm: Any, eval_emb: Any
) -> Tuple[List[Any], List[str], List[Dict[str, Any]]]:
    skipped: List[Dict[str, Any]] = []
    metrics: List[Any] = []
    names: List[str] = []

    try:
        from ragas.llms import LangchainLLMWrapper  # type: ignore

        wrapped_llm = LangchainLLMWrapper(eval_llm)
    except Exception as e:
        wrapped_llm = eval_llm
        skipped.append(
            {
                "metric": "__wrapper_llm__",
                "reason": f"LangchainLLMWrapper_unavailable: {type(e).__name__}",
            }
        )

    try:
        from ragas.embeddings import LangchainEmbeddingsWrapper  # type: ignore

        wrapped_emb = LangchainEmbeddingsWrapper(eval_emb)
    except Exception as e:
        wrapped_emb = eval_emb
        skipped.append(
            {
                "metric": "__wrapper_emb__",
                "reason": f"LangchainEmbeddingsWrapper_unavailable: {type(e).__name__}",
            }
        )

    try:
        faithfulness_obj, relevancy_obj = _import_metric_classes()
    except Exception as e:
        skipped.append(
            {
                "metric": "__metric_import__",
                "reason": f"{type(e).__name__}: {e}",
            }
        )
        return metrics, names, skipped

    try:
        if isinstance(faithfulness_obj, type):
            m_faith = faithfulness_obj(llm=wrapped_llm)
        else:
            m_faith = faithfulness_obj
            if hasattr(m_faith, "llm"):
                m_faith.llm = wrapped_llm

        metrics.append(m_faith)
        names.append("faithfulness")
    except Exception as e:
        skipped.append(
            {
                "metric": "faithfulness",
                "reason": f"{type(e).__name__}: {e}",
            }
        )

    try:
        if isinstance(relevancy_obj, type):
            try:
                m_rel = relevancy_obj(
                    llm=wrapped_llm,
                    embeddings=wrapped_emb,
                    strictness=1,
                )
            except TypeError:
                m_rel = relevancy_obj(
                    llm=wrapped_llm,
                    embeddings=wrapped_emb,
                )
                if hasattr(m_rel, "strictness"):
                    m_rel.strictness = 1
        else:
            m_rel = relevancy_obj
            if hasattr(m_rel, "llm"):
                m_rel.llm = wrapped_llm
            if hasattr(m_rel, "embeddings"):
                m_rel.embeddings = wrapped_emb
            if hasattr(m_rel, "strictness"):
                m_rel.strictness = 1

        metrics.append(m_rel)
        names.append("answer_relevancy")
    except Exception as e:
        skipped.append(
            {
                "metric": "answer_relevancy",
                "reason": f"{type(e).__name__}: {e}",
            }
        )

    return metrics, names, skipped


def _is_mistral_message_order_error(exc: Exception) -> bool:
    if not isinstance(exc, ModelHTTPError):
        return False

    body = getattr(exc, "body", "") or ""
    status_code = getattr(exc, "status_code", None)

    return status_code == 400 and (
        "invalid_request_message_order" in body
        or "Expected last role User or Tool" in body
        or "got assistant" in body
        or '"code":"3230"' in body
    )


def run_agent_with_retry(
    *,
    user_question: UserQuestion,
    vector_store_manager: VectorStoreManager,
    default_top_k: int,
    max_retries: int,
    base_sleep: float,
):
    """
    Exécute run_agent_with_trace avec retry/backoff sur :
    - 429 Too Many Requests
    - 400 invalid_request_message_order (Mistral)
    """
    attempt = 0
    max_attempts = max_retries + 1
    last_exc: Exception | None = None

    while attempt < max_attempts:
        try:
            return run_agent_with_trace(
                user_question=user_question,
                vector_store_manager=vector_store_manager,
                default_top_k=default_top_k,
            )

        except ModelHTTPError as e:
            last_exc = e
            status_code = getattr(e, "status_code", None)
            sleep_s = base_sleep * (2**attempt)

            if status_code == 429:
                if attempt >= max_retries:
                    raise

                print(
                    f"[agent retry] 429 reçu sur la question: {user_question.question[:80]!r} "
                    f"-> pause {sleep_s:.1f}s (tentative {attempt + 1}/{max_attempts})"
                )
                time.sleep(sleep_s)
                attempt += 1
                continue

            if _is_mistral_message_order_error(e):
                if attempt >= max_retries:
                    raise

                print(
                    f"[agent retry] message order Mistral sur la question: {user_question.question[:80]!r} "
                    f"-> relance complète dans {sleep_s:.1f}s "
                    f"(tentative {attempt + 1}/{max_attempts})"
                )
                time.sleep(sleep_s)
                attempt += 1
                continue

            raise

        except Exception as e:
            last_exc = e
            raise

    if last_exc is not None:
        raise last_exc

    raise RuntimeError("run_agent_with_retry a échoué sans exception explicite.")


# -----------------------------
# Main
# -----------------------------
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="evals/experiments")
    parser.add_argument("--k", type=int, default=SEARCH_K)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--use-question-en", action="store_true")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--max-retries", type=int, default=10)
    parser.add_argument("--max-wait", type=int, default=60)
    parser.add_argument("--agent-max-retries", type=int, default=6)
    parser.add_argument("--agent-base-sleep", type=float, default=3.0)
    parser.add_argument("--sleep-between-rows", type=float, default=0.5)
    parser.add_argument(
        "--raise-ragas-exceptions",
        action="store_true",
        help="Active raise_exceptions=True pour débugger RAGAS",
    )
    args = parser.parse_args()

    load_dotenv()
    if not MISTRAL_API_KEY:
        print("ERROR: MISTRAL_API_KEY missing. Set it in .env")
        return 2

    vsm = VectorStoreManager()
    if vsm.index is None or not vsm.document_chunks:
        print("ERROR: Vector store not loaded. Run: uv run python indexer.py")
        return 3

    rows = read_jsonl(Path(args.dataset))
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    run_id = now_tag()
    out_dir = Path(args.out_dir) / f"run_{run_id}"
    ensure_dir(out_dir)

    traces: List[Dict[str, Any]] = []
    dict_samples: List[Dict[str, Any]] = []

    for item in rows:
        q = item.get("question", "")
        if args.use_question_en and item.get("question_en"):
            q = item["question_en"]

        expected_sources = item.get("expected_source_ids") or []
        reference_answer = item.get("reference_answer")
        expected_tool = item.get("expected_tool")

        user_question = UserQuestion(question=q, top_k=args.k)
        final_answer, trace = run_agent_with_retry(
            user_question=user_question,
            vector_store_manager=vsm,
            default_top_k=args.k,
            max_retries=args.agent_max_retries,
            base_sleep=args.agent_base_sleep,
        )

        answer = extract_answer_text(final_answer)
        sources = extract_sources(final_answer)
        notes = extract_notes(final_answer)
        used_retrieval_flag = extract_used_retrieval(final_answer)

        used_tools = trace.get("used_tools") or []
        retrieved_sources_ranked = trace.get("retrieved_source_ids") or []
        contexts = build_eval_contexts_from_trace(trace)

        rmet = retrieval_metrics(expected_sources, retrieved_sources_ranked)
        tmet = tool_routing_metrics(expected_tool, used_tools)

        trace_row = {
            "id": item.get("id"),
            "category": item.get("category"),
            "lang": item.get("lang"),
            "question": q,
            "expected_tool": expected_tool,
            "used_tools": used_tools,
            "answer": answer,
            "final_sources": sources,
            "final_notes": notes,
            "final_used_retrieval": used_retrieval_flag,
            "retrieved_contexts": contexts,
            "retrieved_source_ids": retrieved_sources_ranked,
            "expected_source_ids": expected_sources,
            "reference_answer": reference_answer,
            "sql_query": trace.get("sql_query"),
            "sql_row_count": trace.get("sql_row_count"),
            "sql_rows_preview": trace.get("sql_rows_preview"),
            "sql_notes": trace.get("sql_notes"),
            **rmet,
            **tmet,
        }

        traces.append(trace_row)

        dict_samples.append(
            {
                "user_input": q,
                "response": answer,
                "retrieved_contexts": contexts,
                "reference": reference_answer,
                "question": q,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": reference_answer,
            }
        )

        if args.sleep_between_rows > 0:
            time.sleep(args.sleep_between_rows)

    traces_path = out_dir / "traces.jsonl"
    with traces_path.open("w", encoding="utf-8") as f:
        for t in traces:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    eval_llm = SafeChatMistralAI(
        model=MODEL_NAME,
        temperature=0,
        mistral_api_key=MISTRAL_API_KEY,
    )
    eval_emb = MistralAIEmbeddings(
        model="mistral-embed",
        mistral_api_key=MISTRAL_API_KEY,
    )

    core_metrics, metrics_loaded, metrics_skipped = build_core_ragas_metrics(
        eval_llm, eval_emb
    )

    import pandas as pd

    base_df = pd.DataFrame(traces)
    ragas_df = None

    if core_metrics:
        eval_ds = EvaluationDataset.from_list(dict_samples)

        eval_kwargs = {
            "dataset": eval_ds,
            "metrics": core_metrics,
            "raise_exceptions": args.raise_ragas_exceptions,
        }

        if RunConfig is not None:
            eval_kwargs["run_config"] = RunConfig(
                max_workers=args.max_workers,
                max_retries=args.max_retries,
                max_wait=args.max_wait,
            )

        result = evaluate(**eval_kwargs)
        ragas_df = result.to_pandas()

    if ragas_df is not None:
        for col in base_df.columns:
            ragas_df[col] = base_df[col].values
        final_df = ragas_df
    else:
        final_df = base_df

    # placeholders restant pour plus tard
    final_df["sql_query_equivalence_ragas"] = None
    final_df["datacompy_score_ragas"] = None

    out_csv = out_dir / "results.csv"
    final_df.to_csv(out_csv, index=False, encoding="utf-8")

    summary = {
        "run_id": run_id,
        "dataset": str(Path(args.dataset)),
        "k": args.k,
        "model_name": MODEL_NAME,
        "metrics_loaded": metrics_loaded,
        "metrics_skipped": metrics_skipped,
        "notes": [
            "Cette version évalue l'agent hybride via run_agent_with_trace(...), pas l'ancien pipeline direct.",
            "Les retrieved_contexts fournis à RAGAS incluent maintenant les chunks RAG et un résumé de preuve SQL quand le tool SQL est utilisé.",
            "Les métriques de tool calling sont calculées à partir de expected_tool vs used_tools.",
            "Faithfulness et answer_relevancy restent évaluées via RAGAS avec le même montage de métriques qu'avant.",
            "Pour réduire les 429: max_workers=1 et retries élevés.",
            "Le runner retente aussi les erreurs Mistral de type invalid_request_message_order.",
        ],
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"OK - saved:\n- {traces_path}\n- {out_csv}\n- {out_dir / 'summary.json'}")
    if metrics_skipped:
        print(
            "WARNING - some metrics were skipped. See summary.json -> metrics_skipped"
        )

    try:
        num_means = (
            final_df.select_dtypes("number")
            .mean(numeric_only=True)
            .sort_values(ascending=False)
        )
        print("\nTop numeric means (quick glance):")
        print(num_means.head(20).to_string())
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
