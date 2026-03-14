from __future__ import annotations

from pathlib import Path
import pandas as pd


OLD_RUN = Path("evals/experiments/run_20260313_144311/results.csv")
NEW_RUN = Path("evals/experiments/run_20260313_160107/results.csv")
OUT_DIR = Path("evals/experiments/comparisons")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def find_key_column(df: pd.DataFrame) -> str:
    candidates = [
        "question",
        "user_input",
        "query",
        "prompt",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(
        f"Aucune colonne de jointure trouvée. Colonnes disponibles: {list(df.columns)}"
    )


def existing_numeric_cols(df: pd.DataFrame, wanted: list[str]) -> list[str]:
    return [c for c in wanted if c in df.columns]


def main() -> None:
    old_df = pd.read_csv(OLD_RUN)
    new_df = pd.read_csv(NEW_RUN)

    key_col_old = find_key_column(old_df)
    key_col_new = find_key_column(new_df)

    if key_col_old != key_col_new:
        new_df = new_df.rename(columns={key_col_new: key_col_old})

    key_col = key_col_old

    interesting_metrics = [
        "faithfulness",
        "answer_relevancy",
        "hit_at_k",
        "mrr",
        "recall_at_k",
        "precision_at_k",
        "tool_routing_ok",
        "tool_call_accuracy_ragas",
        "tool_call_precision_ragas",
        "tool_call_recall_ragas",
        "tool_call_f1_ragas",
        "sql_row_count",
    ]

    old_metrics = existing_numeric_cols(old_df, interesting_metrics)
    new_metrics = existing_numeric_cols(new_df, interesting_metrics)
    common_metrics = [c for c in interesting_metrics if c in old_metrics and c in new_metrics]

    useful_text_cols = [
        "expected_tool",
        "chosen_tool",
        "tool_name",
        "final_answer",
        "answer",
        "ground_truth",
        "reference",
        "notes",
        "sql_query",
        "retrieved_sources",
    ]
    old_text_cols = [c for c in useful_text_cols if c in old_df.columns]
    new_text_cols = [c for c in useful_text_cols if c in new_df.columns]

    old_keep = [key_col] + common_metrics + old_text_cols
    new_keep = [key_col] + common_metrics + new_text_cols

    merged = old_df[old_keep].merge(
        new_df[new_keep],
        on=key_col,
        how="outer",
        suffixes=("_old", "_new"),
        indicator=True,
    )

    # Deltas numériques
    for col in common_metrics:
        merged[f"delta_{col}"] = merged[f"{col}_new"] - merged[f"{col}_old"]

    # Score synthétique de dégradation
    score_cols = [
        c for c in [
            "faithfulness",
            "answer_relevancy",
            "hit_at_k",
            "mrr",
            "recall_at_k",
            "precision_at_k",
            "tool_routing_ok",
            "tool_call_accuracy_ragas",
        ]
        if f"delta_{c}" in merged.columns
    ]

    merged["degradation_score"] = 0.0
    for col in score_cols:
        merged["degradation_score"] += (-merged[f"delta_{col}"]).fillna(0)

    # Cas dégradés
    degraded = merged.sort_values("degradation_score", ascending=False)

    # Cas améliorés
    improvement_cols = [f"delta_{c}" for c in score_cols]
    improved = merged.copy()
    improved["improvement_score"] = 0.0
    for col in score_cols:
        improved["improvement_score"] += improved[f"delta_{col}"].fillna(0)
    improved = improved.sort_values("improvement_score", ascending=False)

    # Cas potentiellement problématiques
    suspects_mask = False
    for col in [
        "tool_routing_ok_new",
        "tool_call_accuracy_ragas_new",
        "hit_at_k_new",
        "faithfulness_new",
        "answer_relevancy_new",
    ]:
        if col in merged.columns:
            if "faithfulness" in col or "answer_relevancy" in col:
                suspects_mask = suspects_mask | (merged[col] < 0.7)
            else:
                suspects_mask = suspects_mask | (merged[col] < 1.0)

    suspects = merged[suspects_mask].copy()

    merged.to_csv(OUT_DIR / "all_diffs.csv", index=False)
    degraded.head(30).to_csv(OUT_DIR / "top_degradations.csv", index=False)
    improved.head(30).to_csv(OUT_DIR / "top_improvements.csv", index=False)
    suspects.to_csv(OUT_DIR / "suspects.csv", index=False)

    print("Comparaison terminée.")
    print(f"Clé utilisée : {key_col}")
    print(f"Métriques comparées : {common_metrics}")
    print()
    print("Fichiers générés :")
    print(f"- {OUT_DIR / 'all_diffs.csv'}")
    print(f"- {OUT_DIR / 'top_degradations.csv'}")
    print(f"- {OUT_DIR / 'top_improvements.csv'}")
    print(f"- {OUT_DIR / 'suspects.csv'}")


if __name__ == "__main__":
    main()
