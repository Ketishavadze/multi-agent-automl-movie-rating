import pandas as pd
import numpy as np

from config import (
    CLEAN_DATA_PATH,
    ENGINEERED_DATA_PATH,
    FEATURE_REPORT_PATH,
    TARGET_COLUMN,
    RATING_COLUMN,
)

from tools.data_tools import (
    inspect_metadata,
    get_main_genre,
    count_json_items,
    get_director,
    get_main_actor,
    get_season,
)

from llm_client import ask_llm


def safe_mode(series, default_value):
    mode_values = series.dropna().mode()
    if len(mode_values) > 0:
        return mode_values[0]
    return default_value


def run_feature_engineer():
    df = pd.read_csv(CLEAN_DATA_PATH)

    before_metadata = inspect_metadata(df)

    report_lines = []
    report_lines.append("Agent 2 — Feature Engineer")
    report_lines.append(f"Input shape: {df.shape}")

    # Ask LLM for feature engineering plan
    system_prompt = """
You are Agent 2: The Feature Engineer.

You receive a cleaned movie dataset and a cleaning summary.
Your goal is to create useful features for predicting whether a movie is highly rated.

Focus on semantic movie logic:
- release date features
- genre features
- director and cast features
- financial features
- popularity and vote features
- avoiding target leakage

Important:
The target will be is_highly_rated = vote_average >= 7.5.
Do NOT use vote_average as a model feature because that would leak the answer.

Return a concise feature engineering strategy.
"""

    user_prompt = f"""
Cleaned dataset metadata:
{before_metadata}

Available columns:
{df.columns.tolist()}

Task:
Suggest what features should be created and which columns should be removed before model training.
Explain why these choices make sense for movie rating classification.
"""

    try:
        llm_plan = ask_llm(system_prompt, user_prompt)

        report_lines.append("\nLLM Feature Engineering Plan:")
        report_lines.append(llm_plan)

    except Exception as e:
        report_lines.append("\nLLM Feature Engineering Plan:")
        report_lines.append(f"LLM call failed: {e}")

    # Create target variable
    if RATING_COLUMN not in df.columns:
        raise ValueError(f"Missing rating column: {RATING_COLUMN}")

    df[TARGET_COLUMN] = (df[RATING_COLUMN] >= 7.5).astype(int)
    report_lines.append("\nCreated target is_highly_rated using vote_average >= 7.5.")

    # Convert important numeric columns
    for col in ["budget", "revenue", "runtime", "popularity", "vote_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(0)

    # Release date features
    if "release_date" in df.columns:
        df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")

        df["release_year"] = df["release_date"].dt.year
        df["release_month"] = df["release_date"].dt.month

        median_year = df["release_year"].median()
        mode_month = safe_mode(df["release_month"], 1)

        df["release_year"] = df["release_year"].fillna(median_year)
        df["release_month"] = df["release_month"].fillna(mode_month)

        df["movie_age"] = 2026 - df["release_year"]
        df["release_season"] = df["release_month"].apply(get_season)

        report_lines.append(
            "Extracted release_year, release_month, movie_age, and release_season from release_date."
        )

    # JSON/movie semantic features
    if "genres" in df.columns:
        df["main_genre"] = df["genres"].apply(get_main_genre)
        df["genre_count"] = df["genres"].apply(count_json_items)
        report_lines.append("Extracted main_genre and genre_count from genres.")

    if "keywords" in df.columns:
        df["keyword_count"] = df["keywords"].apply(count_json_items)
        report_lines.append("Created keyword_count from keywords.")

    if "production_companies" in df.columns:
        df["company_count"] = df["production_companies"].apply(count_json_items)
        report_lines.append("Created company_count from production_companies.")

    if "production_countries" in df.columns:
        df["country_count"] = df["production_countries"].apply(count_json_items)
        report_lines.append("Created country_count from production_countries.")

    if "spoken_languages" in df.columns:
        df["spoken_language_count"] = df["spoken_languages"].apply(count_json_items)
        report_lines.append("Created spoken_language_count from spoken_languages.")

    # Cast and crew features
    if "crew" in df.columns:
        df["director_name"] = df["crew"].apply(get_director)
        df["crew_count"] = df["crew"].apply(count_json_items)
        df["director_movie_count"] = df.groupby("director_name")["director_name"].transform("count")

        report_lines.append(
            "Extracted director_name, crew_count, and director_movie_count from crew."
        )

    if "cast" in df.columns:
        df["main_actor"] = df["cast"].apply(get_main_actor)
        df["cast_count"] = df["cast"].apply(count_json_items)
        df["main_actor_movie_count"] = df.groupby("main_actor")["main_actor"].transform("count")

        report_lines.append(
            "Extracted main_actor, cast_count, and main_actor_movie_count from cast."
        )

    # Financial features
    if "budget" in df.columns and "revenue" in df.columns:
        df["has_budget"] = (df["budget"] > 0).astype(int)
        df["has_revenue"] = (df["revenue"] > 0).astype(int)

        df["profit"] = df["revenue"] - df["budget"]

        df["profit_ratio"] = np.where(
            df["budget"] > 0,
            df["revenue"] / df["budget"],
            0
        )

        df["log_budget"] = np.log1p(df["budget"])
        df["log_revenue"] = np.log1p(df["revenue"])

        report_lines.append(
            "Created has_budget, has_revenue, profit, profit_ratio, log_budget, and log_revenue."
        )

    # Avoid high-cardinality categorical explosion
    # We use frequency features for director/main_actor, then drop raw names.
    high_cardinality_cols = [
        "director_name",
        "main_actor",
    ]

    existing_high_cardinality_cols = [
        col for col in high_cardinality_cols if col in df.columns
    ]

    if existing_high_cardinality_cols:
        df = df.drop(columns=existing_high_cardinality_cols)
        report_lines.append(
            f"Dropped high-cardinality name columns after extracting frequency features: {existing_high_cardinality_cols}"
        )

    # Drop leakage and raw text/JSON columns
    drop_cols = [
        RATING_COLUMN,          # target leakage
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
    ]

    existing_drop_cols = [col for col in drop_cols if col in df.columns]

    if existing_drop_cols:
        df = df.drop(columns=existing_drop_cols)
        report_lines.append(f"Dropped raw/leakage columns: {existing_drop_cols}")

    # One-hot encode only remaining low-cardinality categorical columns
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        report_lines.append(f"One-hot encoded categorical columns: {categorical_cols}")
    else:
        report_lines.append("No categorical columns remained for one-hot encoding.")

    # Final cleanup
    df = df.replace([np.inf, -np.inf], 0)
    df = df.fillna(0)

    # Make sure target is last column
    if TARGET_COLUMN in df.columns:
        target = df[TARGET_COLUMN]
        df = df.drop(columns=[TARGET_COLUMN])
        df[TARGET_COLUMN] = target

    after_metadata = inspect_metadata(df)

    report_lines.append("\nMetadata after feature engineering:")
    report_lines.append(str(after_metadata))

    df.to_csv(ENGINEERED_DATA_PATH, index=False)

    report_lines.append(f"\nOutput shape: {df.shape}")
    report_lines.append(f"Saved engineered data to {ENGINEERED_DATA_PATH}")

    with open(FEATURE_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    return "\n".join(report_lines)