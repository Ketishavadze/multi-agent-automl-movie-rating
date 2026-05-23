import pandas as pd

from config import RAW_DATA_PATH, CLEAN_DATA_PATH, CLEANING_REPORT_PATH
from tools.data_tools import inspect_metadata
from llm_client import ask_llm


def run_data_cleaner():
    df = pd.read_csv(RAW_DATA_PATH)

    before_metadata = inspect_metadata(df)

    report_lines = []
    report_lines.append("Agent 1 — Data Cleaner")
    report_lines.append(f"Original shape: {df.shape}")

    # Ask LLM for cleaning plan
    system_prompt = """
You are Agent 1: The Data Cleaner.
You inspect movie dataset metadata and explain which cleaning actions should be taken.
Return a concise action plan.
"""

    user_prompt = f"""
Dataset metadata:
{before_metadata}

Target later will be is_highly_rated created from vote_average >= 7.5.

Explain what should be cleaned and why.
"""

    try:
        llm_plan = ask_llm(system_prompt, user_prompt)

        report_lines.append("\nLLM Cleaning Plan:")
        report_lines.append(llm_plan)

    except Exception as e:
        report_lines.append("\nLLM Cleaning Plan:")
        report_lines.append(f"LLM call failed: {e}")

    # Create has_homepage before dropping homepage
    if "homepage" in df.columns:
        df["has_homepage"] = df["homepage"].notna().astype(int)
        report_lines.append("Created has_homepage from homepage column.")

    # Drop pure identifier / duplicate columns
    columns_to_drop = []

    for col in ["id", "movie_id", "homepage"]:
        if col in df.columns:
            columns_to_drop.append(col)

    # title_y appears after merge sometimes
    if "title_y" in df.columns:
        columns_to_drop.append("title_y")

    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)
        report_lines.append(f"Dropped columns: {columns_to_drop}")

    # Rename title_x if it exists
    if "title_x" in df.columns:
        df = df.rename(columns={"title_x": "title"})
        report_lines.append("Renamed title_x to title.")

    # Convert release_date
    if "release_date" in df.columns:
        df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
        report_lines.append("Converted release_date to datetime.")

    # Fill runtime
    if "runtime" in df.columns:
        median_runtime = df["runtime"].median()
        df["runtime"] = df["runtime"].fillna(median_runtime)
        report_lines.append(f"Imputed missing runtime using median: {median_runtime}")

    # Fill simple categorical missing values
    for col in ["original_language", "status"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")
            report_lines.append(f"Filled missing {col} with Unknown.")

    after_metadata = inspect_metadata(df)

    report_lines.append("\nMetadata after cleaning:")
    report_lines.append(str(after_metadata))

    df.to_csv(CLEAN_DATA_PATH, index=False)

    report_lines.append(f"\nCleaned shape: {df.shape}")
    report_lines.append(f"Saved clean data to {CLEAN_DATA_PATH}")

    with open(CLEANING_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    return "\n".join(report_lines)