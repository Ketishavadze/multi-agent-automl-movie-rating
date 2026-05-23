# Multi-Agent Movie AutoML

## Overview

This project implements a three-agent LLM AutoML pipeline for movie rating classification.
The system predicts whether a movie is **highly rated** using TMDB-style movie metadata.

A movie is labeled as highly rated if:

```python
is_highly_rated = 1 if vote_average >= 7.5 else 0
```

The project follows a sequential multi-agent workflow:

```
Raw Movie Dataset
      ↓
Agent 1: Data Cleaner
      ↓
clean_data.csv + cleaning_report.txt
      ↓
Agent 2: Feature Engineer
      ↓
engineered_data.csv + feature_report.txt
      ↓
Agent 3: Model Trainer
      ↓
XGBoost training loop + model_logs.txt
      ↓
final_report.md
```

The goal is not only to train a model, but to show how LLM agents can make data science decisions, generate code, evaluate results, and improve the model through feedback.

---

## Project Topic

**Predicting Whether a Movie Becomes Highly Rated**

The system uses metadata such as:

- `budget`
- `revenue`
- `runtime`
- `popularity`
- `vote count`
- `release date`
- `genres`
- `keywords`
- `production companies`
- `production countries`
- `spoken languages`
- `cast`
- `crew`
- `director information`
- `main actor information`

The target column is `is_highly_rated`, where:

- `1` = `vote_average >= 7.5`
- `0` = `vote_average < 7.5`

> The original `vote_average` column is removed before training to avoid target leakage.

---

## Agents

### Agent 1: Data Cleaner

The Data Cleaner inspects the raw merged movie dataset and prepares it for feature engineering.

**Main responsibilities:**

- Inspect dataset shape, data types, null counts, and unique counts
- Identify missing values
- Detect identifier columns
- Convert `release_date` to datetime
- Impute missing `runtime` values
- Create `has_homepage`
- Remove duplicated or unusable columns

**Example decisions:**

- Dropped `id` and `movie_id` because they are identifiers
- Dropped `homepage` after extracting `has_homepage`
- Renamed `title_x` to `title`
- Converted `release_date` to datetime
- Imputed missing `runtime` using the median

**Output:**

```
outputs/clean_data.csv
outputs/cleaning_report.txt
```

---

### Agent 2: Feature Engineer

The Feature Engineer receives the cleaned dataset and creates semantic movie-specific features.

**Created features include:**

| Feature | Description |
|---|---|
| `release_year` | Year of release |
| `release_month` | Month of release |
| `movie_age` | Age of movie from current year |
| `release_season` | Season derived from release month |
| `main_genre` | Primary genre |
| `genre_count` | Number of genres |
| `keyword_count` | Number of keywords |
| `company_count` | Number of production companies |
| `country_count` | Number of production countries |
| `spoken_language_count` | Number of spoken languages |
| `director_movie_count` | Director's total movie count |
| `main_actor_movie_count` | Main actor's total movie count |
| `cast_count` | Cast size |
| `crew_count` | Crew size |
| `profit` | Revenue minus budget |
| `profit_ratio` | Profit relative to budget |
| `log_budget` | Log-transformed budget |
| `log_revenue` | Log-transformed revenue |
| `has_budget` | Whether budget data exists |
| `has_revenue` | Whether revenue data exists |

**Removed columns (raw text, JSON-like, or leakage):**

- `vote_average`, `genres`, `keywords`, `cast`, `crew`
- `overview`, `tagline`, `title`, `original_title`, `release_date`

**Output:**

```
outputs/engineered_data.csv
outputs/feature_report.txt
```

---

### Agent 3: Model Trainer

The Model Trainer uses an LLM to generate executable Python code for XGBoost training.

**Feedback loop:**

1. Generate baseline XGBoost code
2. Execute the generated code
3. Read accuracy, recall, and F1 score
4. Decide whether the model is good enough
5. If not, generate improved code with adjusted hyperparameters
6. Stop after a successful result or after the maximum number of attempts

**Output:**

```
outputs/model_logs.txt
outputs/final_report.md
```

---

## Model Results

The model was trained to classify movies as highly rated or not highly rated.

| Attempt | Accuracy | Recall | F1 Score | Decision |
|---|---|---|---|---|
| 1 | 0.9469 | 0.4853 | 0.5641 | Continue |
| 2 | **0.9521** | **0.5373** | **0.6102** | Continue |
| 3 | 0.9500 | 0.5224 | 0.5932 | Stop |

**Best result: Attempt 2** — highest accuracy, recall, and F1 score.

Although accuracy is high, recall is lower because highly rated movies are likely a minority class. The model is strong at classifying most movies overall, but still misses some truly highly rated movies.

---

## Project Structure

```
multi-agent-movie-automl/
│
├── agents/
│   ├── data_cleaner.py
│   ├── feature_engineer.py
│   └── model_trainer.py
│
├── tools/
│   ├── data_tools.py
│   └── code_executor.py
│
├── data/
│   ├── tmdb_5000_movies.csv
│   ├── tmdb_5000_credits.csv
│   └── merged_movies.csv
│
├── outputs/
│   ├── clean_data.csv
│   ├── engineered_data.csv
│   ├── cleaning_report.txt
│   ├── feature_report.txt
│   ├── model_logs.txt
│   └── final_report.md
│
├── config.py
├── llm_client.py
├── main.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/multi-agent-movie-automl.git
cd multi-agent-movie-automl
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

```bash
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your OpenAI API key

Create a `.env` file:

```env
OPENAI_API_KEY=your_api_key_here
```

> **Do not commit `.env` to GitHub.**

---

## Running the Project

Run the full pipeline:

```bash
python main.py
```

This will run all three agents sequentially:

1. **Agent 1:** Data Cleaner
2. **Agent 2:** Feature Engineer
3. **Agent 3:** Model Trainer

After execution, all generated files will appear in the `outputs/` folder.

---

## Requirements

```
pandas
numpy
scikit-learn
xgboost
openai
python-dotenv
```

---

## Important Notes

- The LLM does **not** directly predict movie ratings
- LLM agents make decisions, generate plans, and write training code
- The actual prediction is done by the **XGBoost model**
- The model predicts a binary label: **highly rated** or **not highly rated**
- `vote_average` is removed before training to prevent target leakage
- The feedback loop in Agent 3 demonstrates iterative model improvement

---

## Assignment Checklist

- [x] Agent 1 cleans the data and passes it to Agent 2
- [x] Agent 2 creates new features using movie-specific logic
- [x] Agent 3 generates and executes Python code
- [x] Agent 3 reacts to model results through a feedback loop
- [x] Final logs and reports are saved in the `outputs/` folder