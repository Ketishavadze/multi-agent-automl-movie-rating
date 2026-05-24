import ast
import json
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def inspect_metadata(df: pd.DataFrame) -> Dict[str, Any]:
    """Return compact metadata that LLM agents can reason over."""
    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "null_counts": df.isnull().sum().to_dict(),
        "null_percentages": (df.isnull().mean() * 100).round(2).to_dict(),
        "unique_counts": df.nunique(dropna=True).to_dict(),
    }


def get_column_stats(df: pd.DataFrame, col: str) -> Dict[str, Any]:
    """Tool required by the assignment: summarize one column for an agent."""
    if col not in df.columns:
        return {"error": f"Column '{col}' not found."}

    series = df[col]
    stats: Dict[str, Any] = {
        "column": col,
        "dtype": str(series.dtype),
        "null_count": int(series.isna().sum()),
        "null_percentage": round(float(series.isna().mean() * 100), 2),
        "unique_count": int(series.nunique(dropna=True)),
    }

    if pd.api.types.is_numeric_dtype(series):
        desc = series.describe().replace({np.nan: None}).to_dict()
        stats["numeric_summary"] = {k: float(v) if v is not None else None for k, v in desc.items()}
    else:
        stats["top_values"] = series.fillna("<MISSING>").astype(str).value_counts().head(10).to_dict()

    return stats


def impute_missing(df: pd.DataFrame, col: str, strategy: str) -> pd.DataFrame:
    """Tool required by the assignment: impute missing values with an agent-selected strategy."""
    if col not in df.columns:
        return df

    strategy = strategy.lower().strip()
    if strategy == "mean" and pd.api.types.is_numeric_dtype(df[col]):
        value = df[col].mean()
    elif strategy == "median" and pd.api.types.is_numeric_dtype(df[col]):
        value = df[col].median()
    elif strategy == "mode":
        mode_values = df[col].dropna().mode()
        value = mode_values.iloc[0] if len(mode_values) else "Unknown"
    elif strategy in {"unknown", "constant_unknown"}:
        value = "Unknown"
    elif strategy in {"zero", "constant_zero"}:
        value = 0
    else:
        # Safe fallback: numeric columns get median, non-numeric columns get Unknown.
        value = df[col].median() if pd.api.types.is_numeric_dtype(df[col]) else "Unknown"

    df[col] = df[col].fillna(value)
    return df


def drop_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Tool required by the assignment: drop one unusable column."""
    if col in df.columns:
        return df.drop(columns=[col])
    return df


def create_interaction(df: pd.DataFrame, expression: str, new_column: Optional[str] = None) -> pd.DataFrame:
    """Create a simple math feature from a safe pandas expression.

    Example:
        create_interaction(df, "revenue / budget", "revenue_per_budget")
    """
    if not new_column:
        new_column = expression.replace(" ", "_").replace("/", "_per_").replace("*", "_x_")

    safe_locals = {col: df[col] for col in df.columns}
    safe_locals.update({"np": np, "log1p": np.log1p})
    try:
        df[new_column] = pd.eval(expression, local_dict=safe_locals, engine="python")
        df[new_column] = df[new_column].replace([np.inf, -np.inf], 0).fillna(0)
    except Exception:
        # Do not crash the agent pipeline if the LLM proposes a bad expression.
        pass
    return df


def encode_categorical(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Tool required by the assignment: one-hot encode a categorical column."""
    if col in df.columns:
        return pd.get_dummies(df, columns=[col], drop_first=True)
    return df


def correlation_analysis(df: pd.DataFrame, target: str) -> pd.Series:
    """Return absolute correlation with the target for numeric columns."""
    if target not in df.columns:
        return pd.Series(dtype=float)
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    if target not in numeric_df.columns:
        return pd.Series(dtype=float)
    corr = numeric_df.corr(numeric_only=True)[target].drop(labels=[target], errors="ignore")
    return corr.abs().sort_values(ascending=False)


def remove_highly_correlated_features(
    df: pd.DataFrame,
    target: str,
    threshold: float = 0.95,
) -> tuple[pd.DataFrame, List[str]]:
    """Remove redundant numeric features while keeping the one more related to the target."""
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target]
    if len(numeric_cols) < 2:
        return df, []

    target_corr = correlation_analysis(df, target)
    feature_corr = df[numeric_cols].corr(numeric_only=True).abs()
    upper = feature_corr.where(np.triu(np.ones(feature_corr.shape), k=1).astype(bool))

    to_drop = set()
    for col in upper.columns:
        correlated = upper.index[upper[col] > threshold].tolist()
        for row_col in correlated:
            col_score = float(target_corr.get(col, 0))
            row_score = float(target_corr.get(row_col, 0))
            loser = col if col_score < row_score else row_col
            to_drop.add(loser)

    existing = [c for c in to_drop if c in df.columns]
    if existing:
        df = df.drop(columns=existing)
    return df, existing


def select_top_features(df: pd.DataFrame, target: str, k: int = 40) -> pd.DataFrame:
    """Tool required by the assignment: keep the k most target-correlated numeric features."""
    if target not in df.columns:
        return df

    k = max(1, int(k))
    scores = correlation_analysis(df, target)
    keep = scores.head(k).index.tolist()
    keep.append(target)
    keep = [c for c in keep if c in df.columns]
    return df[keep]


def parse_json_list(value: Any) -> list:
    try:
        if pd.isna(value):
            return []
        if isinstance(value, list):
            return value
        return ast.literal_eval(str(value))
    except Exception:
        return []


def extract_names_from_json(value: Any) -> List[str]:
    items = parse_json_list(value)
    names: List[str] = []
    for item in items:
        if isinstance(item, dict) and "name" in item:
            names.append(str(item["name"]))
    return names


def get_main_genre(value: Any) -> str:
    names = extract_names_from_json(value)
    return names[0] if names else "Unknown"


def count_json_items(value: Any) -> int:
    return len(parse_json_list(value))


def get_director(crew_value: Any) -> str:
    crew = parse_json_list(crew_value)
    for person in crew:
        if isinstance(person, dict) and person.get("job") == "Director":
            return str(person.get("name", "Unknown"))
    return "Unknown"


def get_main_actor(cast_value: Any) -> str:
    cast = parse_json_list(cast_value)
    if cast and isinstance(cast[0], dict):
        return str(cast[0].get("name", "Unknown"))
    return "Unknown"


def get_season(month: Any) -> str:
    try:
        month = int(month)
    except Exception:
        return "unknown"

    if month in [12, 1, 2]:
        return "winter"
    if month in [3, 4, 5]:
        return "spring"
    if month in [6, 7, 8]:
        return "summer"
    return "fall"


def extract_json_object(text: str) -> Dict[str, Any]:
    """Extract a JSON object from an LLM response, even if it used markdown fences."""
    if not text:
        return {}
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}

    try:
        return json.loads(cleaned[start : end + 1])
    except Exception:
        return {}
