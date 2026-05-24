import json
from typing import Any, Dict, List

import pandas as pd

from config import CLEAN_DATA_PATH, CLEANING_REPORT_PATH, RAW_DATA_PATH
from llm_client import ask_llm
from tools.data_tools import (
    drop_column,
    extract_json_object,
    get_column_stats,
    impute_missing,
    inspect_metadata,
)


def _fallback_cleaning_plan(df: pd.DataFrame) -> Dict[str, Any]:
    """Used only if the LLM fails or returns invalid JSON."""
    drop_candidates: List[str] = []
    rename_columns: Dict[str, str] = {}
    datetime_columns: List[str] = []
    missing_rules: List[Dict[str, str]] = []
    indicators: List[str] = []

    for col in ["id", "movie_id", "homepage", "title_y"]:
        if col in df.columns:
            drop_candidates.append(col)

    if "title_x" in df.columns:
        rename_columns["title_x"] = "title"

    if "release_date" in df.columns:
        datetime_columns.append("release_date")

    if "runtime" in df.columns and df["runtime"].isna().sum() > 0:
        missing_rules.append({"column": "runtime", "strategy": "median", "reason": "Runtime is numeric and median is robust to outliers."})

    for col in ["original_language", "status"]:
        if col in df.columns and df[col].isna().sum() > 0:
            missing_rules.append({"column": col, "strategy": "unknown", "reason": "Categorical missing value should be explicit."})

    if "homepage" in df.columns:
        indicators.append("homepage")

    return {
        "reason_summary": "Fallback plan used because the LLM response was unavailable or invalid.",
        "create_missing_indicators": indicators,
        "rename_columns": rename_columns,
        "datetime_columns": datetime_columns,
        "missing_value_rules": missing_rules,
        "drop_columns": drop_candidates,
    }


def _request_llm_cleaning_plan(df: pd.DataFrame, metadata: Dict[str, Any]) -> tuple[Dict[str, Any], str]:
    sample_stats = {
        col: get_column_stats(df, col)
        for col in df.columns[:20]
    }

    system_prompt = """
You are Agent 1: The Data Cleaner, also called The Auditor.
You must inspect metadata and return a machine-readable cleaning plan.
The LLM must choose the cleaning actions; do not describe generic advice only.

Return ONLY valid JSON with this schema:
{
  "reason_summary": "short explanation",
  "create_missing_indicators": ["column_name"],
  "rename_columns": {"old_name": "new_name"},
  "datetime_columns": ["column_name"],
  "missing_value_rules": [
    {"column": "column_name", "strategy": "mean|median|mode|unknown|zero", "reason": "why"}
  ],
  "drop_columns": ["column_name"]
}

Rules:
- Preserve vote_average for now; Agent 2 will create the target from it and then remove it.
- Drop pure identifiers and duplicated merge columns.
- If a column contains useful missingness information, request a missing indicator before dropping or imputing it.
- Prefer median for skewed numeric columns, mode/unknown for categorical columns.
"""

    user_prompt = f"""
Dataset metadata:
{json.dumps(metadata, indent=2, default=str)}

Column statistics sample:
{json.dumps(sample_stats, indent=2, default=str)}

Return the cleaning plan as JSON only.
"""

    llm_text = ask_llm(system_prompt, user_prompt)
    plan = extract_json_object(llm_text)
    return plan, llm_text


def run_data_cleaner() -> str:
    df = pd.read_csv(RAW_DATA_PATH)
    before_metadata = inspect_metadata(df)

    report_lines: List[str] = []
    report_lines.append("Agent 1 — Data Cleaner")
    report_lines.append("Goal: audit the raw dataset and prepare clean_data.csv for Agent 2.")
    report_lines.append(f"Original shape: {df.shape}")

    try:
        plan, raw_llm_plan = _request_llm_cleaning_plan(df, before_metadata)
        if not plan:
            plan = _fallback_cleaning_plan(df)
            raw_llm_plan = "LLM returned invalid JSON; fallback plan was used."
    except Exception as exc:
        plan = _fallback_cleaning_plan(df)
        raw_llm_plan = f"LLM call failed: {exc}. Fallback plan was used."

    report_lines.append("\nLLM Cleaning Plan:")
    report_lines.append(raw_llm_plan)
    report_lines.append("\nParsed Cleaning Plan Applied:")
    report_lines.append(json.dumps(plan, indent=2, default=str))

    # 1. Create explicit missing-value indicators selected by the LLM.
    for col in plan.get("create_missing_indicators", []):
        if col in df.columns:
            indicator_col = f"has_{col}"
            df[indicator_col] = df[col].notna().astype(int)
            report_lines.append(f"Created {indicator_col} from missingness in {col}.")

    # 2. Rename columns selected by the LLM.
    valid_renames = {
        old: new
        for old, new in plan.get("rename_columns", {}).items()
        if old in df.columns and isinstance(new, str) and new
    }
    if valid_renames:
        df = df.rename(columns=valid_renames)
        report_lines.append(f"Renamed columns: {valid_renames}")

    # 3. Convert date columns selected by the LLM.
    for col in plan.get("datetime_columns", []):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            report_lines.append(f"Converted {col} to datetime.")

    # 4. Impute columns with LLM-selected strategies.
    for rule in plan.get("missing_value_rules", []):
        col = rule.get("column")
        strategy = rule.get("strategy", "mode")
        if col in df.columns:
            before_nulls = int(df[col].isna().sum())
            df = impute_missing(df, col, strategy)
            after_nulls = int(df[col].isna().sum())
            reason = rule.get("reason", "No reason provided.")
            report_lines.append(
                f"Imputed {col} using {strategy}; nulls {before_nulls} → {after_nulls}. Reason: {reason}"
            )

    # 5. Drop columns selected by the LLM.
    actually_dropped: List[str] = []
    for col in plan.get("drop_columns", []):
        if col in df.columns:
            df = drop_column(df, col)
            actually_dropped.append(col)
    if actually_dropped:
        report_lines.append(f"Dropped columns selected by the LLM: {actually_dropped}")

    after_metadata = inspect_metadata(df)
    df.to_csv(CLEAN_DATA_PATH, index=False)

    report_lines.append("\nMetadata after cleaning:")
    report_lines.append(json.dumps(after_metadata, indent=2, default=str))
    report_lines.append(f"\nCleaned shape: {df.shape}")
    report_lines.append(f"Saved clean data to {CLEAN_DATA_PATH}")

    with open(CLEANING_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    return "\n".join(report_lines)
