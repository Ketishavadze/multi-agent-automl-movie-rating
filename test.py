import pandas as pd

# Load both datasets
movies = pd.read_csv("data/tmdb_5000_movies.csv")
credits = pd.read_csv("data/tmdb_5000_credits.csv")

print("Movies shape:", movies.shape)
print("Credits shape:", credits.shape)

print("\nMovies columns:")
print(movies.columns.tolist())

print("\nCredits columns:")
print(credits.columns.tolist())

# Merge movies with credits
df = movies.merge(
    credits,
    left_on="id",
    right_on="movie_id",
    how="left"
)

print("\nMerged shape:", df.shape)

print("\nMerged columns:")
print(df.columns.tolist())

# Save merged file
df.to_csv("data/merged_movies.csv", index=False)

print("\nMerged dataset saved as data/merged_movies.csv")

import ast

def get_director(crew_str):
    try:
        crew = ast.literal_eval(crew_str)
        for person in crew:
            if person.get("job") == "Director":
                return person.get("name")
        return "Unknown"
    except:
        return "Unknown"


def get_cast_count(cast_str):
    try:
        cast = ast.literal_eval(cast_str)
        return len(cast)
    except:
        return 0


def get_main_actor(cast_str):
    try:
        cast = ast.literal_eval(cast_str)
        if len(cast) > 0:
            return cast[0].get("name", "Unknown")
        return "Unknown"
    except:
        return "Unknown"


df["director_name"] = df["crew"].apply(get_director)
df["cast_count"] = df["cast"].apply(get_cast_count)
df["main_actor"] = df["cast"].apply(get_main_actor)

print(df[["title_x", "director_name", "main_actor", "cast_count"]].head())