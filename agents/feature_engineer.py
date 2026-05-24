import json
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from config import (
    CLEAN_DATA_PATH,
    CLEANING_REPORT_PATH,
    ENGINEERED_DATA_PATH,
    FEATURE_REPORT_PATH,
    RATING_COLUMN,
    TARGET_COLUMN,
)
from llm_client import ask_llm
from tools.data_tools import (
    correlation_analysis,
    count_json_items,
    encode_categorical,
    extract_json_object,
    get_director,
    get_main_actor,
    get_main_genre,
    get_season,
    inspect_metadata,
    remove_highly_correlated_features,
    select_top_features,
)


def _read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "Report not found."


def _fallback_feature_plan(df: pd.DataFrame) -> Dict[str, Any]:
    return {
        "reason_summary": "Fallback feature plan used because the LLM response was unavailable or invalid.",
        "target_threshold": 7.5,
        "feature_operations": [
            "release_date_parts",
            "json_counts",
            "genre_main_category",
            "cast_crew_counts",
            "financial_ratios",
            "log_transforms",
        ],
        "categorical_columns_to_encode": [
            c for c in ["main_genre", "release_season", "original_language"] if c in df.columns or c in ["main_genre", "release_season"]
        ],
        "drop_columns": [
            RATING_COLUMN,
            "genres",
            "keywords",
            "production_companies",
            "production_countries",
            "spoken_languages",
            "cast",
            "crew",
            "overview",
            "tagline",
            "title",
            "original_title",
            "release_date",
            "status",
            "director_name",
            "main_actor",
        ],
        "correlation_threshold": 0.95,
        "top_k_features": 40,
    }


def _request_llm_feature_plan(df: pd.DataFrame, metadata: Dict[str, Any], cleaning_summary: str) -> tuple[Dict[str, Any], str]:
    system_prompt = """
You are Agent 2: The Feature Engineer, also called The Architect.
You receive clean_data.csv and Agent 1's report.
You must create semantic movie features, avoid leakage, encode useful categorical columns, and perform feature selection.

Return ONLY valid JSON with this schema:
{
  "reason_summary": "short explanation",
  "target_threshold": 7.5,
  "feature_operations": [
    "release_date_parts",
    "json_counts",
    "genre_main_category",
    "cast_crew_counts",
    "financial_ratios",
    "log_transforms"
  ],
  "categorical_columns_to_encode": ["column_name"],
  "drop_columns": ["column_name"],
  "correlation_threshold": 0.95,
  "top_k_features": 40
}

Rules:
- The target must be is_highly_rated = 1 if vote_average >= target_threshold else 0.
- vote_average must be removed before model training to avoid target leakage.
- Do not keep raw JSON/text columns after extracting useful information.
- Feature selection is required: choose a top_k_features value and a correlation threshold.
"""

    user_prompt = f"""
Agent 1 cleaning summary:
{cleaning_summary[:5000]}

Cleaned dataset metadata:
{json.dumps(metadata, indent=2, default=str)}

Available columns:
{df.columns.tolist()}

Return the feature-engineering plan as JSON only.
"""

    llm_text = ask_llm(system_prompt, user_prompt)
    plan = extract_json_object(llm_text)
    return plan, llm_text


def run_feature_engineer() -> str:
    df = pd.read_csv(CLEAN_DATA_PATH)
    before_metadata = inspect_metadata(df)
    cleaning_summary = _read_text(CLEANING_REPORT_PATH)

    report_lines: List[str] = []
    report_lines.append("Agent 2 — Feature Engineer")
    report_lines.append("Goal: create semantic movie features and select the most relevant predictors.")
    report_lines.append(f"Input shape: {df.shape}")

    try:
        plan, raw_llm_plan = _request_llm_feature_plan(df, before_metadata, cleaning_summary)
        if not plan:
            plan = _fallback_feature_plan(df)
            raw_llm_plan = "LLM returned invalid JSON; fallback plan was used."
    except Exception as exc:
        plan = _fallback_feature_plan(df)
        raw_llm_plan = f"LLM call failed: {exc}. Fallback plan was used."

    report_lines.append("\nLLM Feature Engineering Plan:")
    report_lines.append(raw_llm_plan)
    report_lines.append("\nParsed Feature Plan Applied:")
    report_lines.append(json.dumps(plan, indent=2, default=str))

    if RATING_COLUMN not in df.columns:
        raise ValueError(f"Missing rating column: {RATING_COLUMN}")

    threshold = float(plan.get("target_threshold", 7.5))
    df[TARGET_COLUMN] = (pd.to_numeric(df[RATING_COLUMN], errors="coerce") >= threshold).astype(int)
    report_lines.append(f"\nCreated target {TARGET_COLUMN} using {RATING_COLUMN} >= {threshold}.")

    for col in ["budget", "revenue", "runtime", "popularity", "vote_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    operations = set(plan.get("feature_operations", []))
    current_year = pd.Timestamp.today().year

    if "release_date_parts" in operations and "release_date" in df.columns:
        df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
        df["release_year"] = df["release_date"].dt.year.fillna(df["release_date"].dt.year.median())
        df["release_month"] = df["release_date"].dt.month.fillna(df["release_date"].dt.month.mode().iloc[0] if not df["release_date"].dt.month.mode().empty else 1)
        df["movie_age"] = current_year - df["release_year"]
        df["release_season"] = df["release_month"].apply(get_season)
        report_lines.append("Created release_year, release_month, movie_age, and release_season from release_date.")

    if "genre_main_category" in operations and "genres" in df.columns:
        df["main_genre"] = df["genres"].apply(get_main_genre)
        report_lines.append("Created main_genre from genres.")

    if "json_counts" in operations:
        json_count_map = {
            "genres": "genre_count",
            "keywords": "keyword_count",
            "production_companies": "company_count",
            "production_countries": "country_count",
            "spoken_languages": "spoken_language_count",
        }
        for source_col, new_col in json_count_map.items():
            if source_col in df.columns:
                df[new_col] = df[source_col].apply(count_json_items)
                report_lines.append(f"Created {new_col} from {source_col}.")

    if "cast_crew_counts" in operations:
        if "crew" in df.columns:
            df["director_name"] = df["crew"].apply(get_director)
            df["crew_count"] = df["crew"].apply(count_json_items)
            df["director_movie_count"] = df.groupby("director_name")["director_name"].transform("count")
            report_lines.append("Created director_name, crew_count, and director_movie_count from crew.")
        if "cast" in df.columns:
            df["main_actor"] = df["cast"].apply(get_main_actor)
            df["cast_count"] = df["cast"].apply(count_json_items)
            df["main_actor_movie_count"] = df.groupby("main_actor")["main_actor"].transform("count")
            report_lines.append("Created main_actor, cast_count, and main_actor_movie_count from cast.")

    if "financial_ratios" in operations and {"budget", "revenue"}.issubset(df.columns):
        df["has_budget"] = (df["budget"] > 0).astype(int)
        df["has_revenue"] = (df["revenue"] > 0).astype(int)
        df["profit"] = df["revenue"] - df["budget"]
        df["profit_ratio"] = np.where(df["budget"] > 0, df["revenue"] / df["budget"], 0)
        report_lines.append("Created has_budget, has_revenue, profit, and profit_ratio.")

    if "log_transforms" in operations:
        for col in ["budget", "revenue", "runtime", "popularity", "vote_count"]:
            if col in df.columns:
                df[f"log_{col}"] = np.log1p(df[col].clip(lower=0))
                report_lines.append(f"Created log_{col}.")

    # Drop leakage/raw columns chosen by the LLM. Always remove vote_average as a safety rule.
    requested_drop_cols = set(plan.get("drop_columns", []))
    requested_drop_cols.add(RATING_COLUMN)
    existing_drop_cols = [c for c in requested_drop_cols if c in df.columns]
    if existing_drop_cols:
        df = df.drop(columns=existing_drop_cols)
        report_lines.append(f"Dropped leakage/raw columns: {existing_drop_cols}")

    # Encode only remaining low-cardinality categorical columns that the LLM selected.
    encoded_cols: List[str] = []
    for col in plan.get("categorical_columns_to_encode", []):
        if col in df.columns and df[col].dtype == "object" and df[col].nunique(dropna=True) <= 30:
            df = encode_categorical(df, col)
            encoded_cols.append(col)
    if encoded_cols:
        report_lines.append(f"One-hot encoded LLM-selected low-cardinality categorical columns: {encoded_cols}")

    # Remove leftover text/object columns that cannot be used by XGBoost directly.
    leftover_object_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if leftover_object_cols:
        df = df.drop(columns=leftover_object_cols)
        report_lines.append(f"Dropped leftover non-numeric columns after feature extraction: {leftover_object_cols}")

    df = df.replace([np.inf, -np.inf], 0).fillna(0)

    # Feature selection required by the assignment.
    before_selection_cols = len(df.columns)
    threshold_corr = float(plan.get("correlation_threshold", 0.95))
    df, redundant_cols = remove_highly_correlated_features(df, TARGET_COLUMN, threshold=threshold_corr)
    if redundant_cols:
        report_lines.append(f"Removed redundant highly correlated features at threshold {threshold_corr}: {redundant_cols}")

    top_k = int(plan.get("top_k_features", 40))
    scores = correlation_analysis(df, TARGET_COLUMN)
    selected_names = scores.head(top_k).index.tolist()
    df = select_top_features(df, TARGET_COLUMN, k=top_k)
    after_selection_cols = len(df.columns)
    report_lines.append(
        f"Selected top {min(top_k, len(selected_names))} predictive features by absolute target correlation. "
        f"Columns reduced from {before_selection_cols} to {after_selection_cols}."
    )
    report_lines.append(f"Top selected features: {selected_names[:20]}")

    if TARGET_COLUMN in df.columns:
        target = df[TARGET_COLUMN].astype(int)
        df = df.drop(columns=[TARGET_COLUMN])
        df[TARGET_COLUMN] = target

    after_metadata = inspect_metadata(df)
    df.to_csv(ENGINEERED_DATA_PATH, index=False)

    report_lines.append("\nMetadata after feature engineering:")
    report_lines.append(json.dumps(after_metadata, indent=2, default=str))
    report_lines.append(f"\nOutput shape: {df.shape}")
    report_lines.append(f"Saved engineered data to {ENGINEERED_DATA_PATH}")

    with open(FEATURE_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    return "\n".join(report_lines)
