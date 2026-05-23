import pandas as pd
import numpy as np
import ast


def inspect_metadata(df):
    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "null_counts": df.isnull().sum().to_dict(),
        "null_percentages": (df.isnull().mean() * 100).round(2).to_dict(),
        "unique_counts": df.nunique().to_dict()
    }


def parse_json_list(value):
    try:
        if pd.isna(value):
            return []
        return ast.literal_eval(value)
    except Exception:
        return []


def extract_names_from_json(value):
    items = parse_json_list(value)
    names = []
    for item in items:
        if isinstance(item, dict) and "name" in item:
            names.append(item["name"])
    return names


def get_main_genre(value):
    names = extract_names_from_json(value)
    return names[0] if len(names) > 0 else "Unknown"


def count_json_items(value):
    return len(parse_json_list(value))


def get_director(crew_value):
    crew = parse_json_list(crew_value)
    for person in crew:
        if isinstance(person, dict) and person.get("job") == "Director":
            return person.get("name", "Unknown")
    return "Unknown"


def get_main_actor(cast_value):
    cast = parse_json_list(cast_value)
    if len(cast) > 0 and isinstance(cast[0], dict):
        return cast[0].get("name", "Unknown")
    return "Unknown"


def get_season(month):
    if month in [12, 1, 2]:
        return "winter"
    elif month in [3, 4, 5]:
        return "spring"
    elif month in [6, 7, 8]:
        return "summer"
    else:
        return "fall"