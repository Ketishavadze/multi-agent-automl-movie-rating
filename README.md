# Multi-Agent Movie AutoML

## Overview

This project implements a three-agent LLM AutoML pipeline for movie rating classification. The system predicts whether a movie is highly rated using TMDB-style movie metadata.

A movie is labeled as highly rated if:

```text
is_highly_rated = 1 if vote_average >= 7.5 else 0
```

The pipeline follows a sequential handoff workflow:

```text
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
XGBoost feedback loop + model_logs.txt
      ↓
final_report.md
```

The goal is not only to train a model, but also to demonstrate how LLM agents can inspect data, make data-science decisions, generate executable code, evaluate results, and improve a model through feedback.

---

## Project Topic

**Predicting Whether a Movie Is Highly Rated**

The system uses movie metadata such as:

- `budget`
- `revenue`
- `runtime`
- `popularity`
- `vote_count`
- `release_date`
- `genres`
- `keywords`
- `production_companies`
- `production_countries`
- `spoken_languages`
- `cast`
- `crew`
- director information
- main actor information

The target column is `is_highly_rated`, where:

- `1` = `vote_average >= 7.5`
- `0` = `vote_average < 7.5`

The original `vote_average` column is removed before model training to avoid direct target leakage.

---

## Agent Architecture

### Agent 1: Data Cleaner — “The Auditor”

The Data Cleaner inspects the raw merged movie dataset and prepares it for feature engineering.

Main responsibilities:

- inspect dataset shape, data types, null counts, null percentages, and unique counts;
- identify missing values, duplicated merge columns, identifier columns, and high-cardinality columns;
- request an LLM-generated cleaning plan in structured JSON;
- apply the parsed cleaning plan using controlled tools;
- save `outputs/clean_data.csv`;
- write `outputs/cleaning_report.txt`.

Example cleaning decisions:

- create `has_homepage` before removing the raw `homepage` URL column;
- drop pure identifiers such as `id` or `movie_id`;
- rename duplicated merge columns such as `title_x`;
- convert `release_date` to a datetime-compatible column;
- impute missing numeric values using an LLM-selected strategy.

### Agent 2: Feature Engineer — “The Architect”

The Feature Engineer receives `clean_data.csv` and Agent 1's cleaning report. It creates movie-specific semantic features and performs feature selection.

Created feature groups include:

| Feature group | Examples |
|---|---|
| Release-date features | `release_year`, `release_month`, `movie_age`, `release_season` |
| Genre features | `main_genre`, `genre_count` |
| JSON-count features | `keyword_count`, `company_count`, `country_count`, `spoken_language_count` |
| Cast/crew features | `director_movie_count`, `main_actor_movie_count`, `cast_count`, `crew_count` |
| Financial features | `has_budget`, `has_revenue`, `profit`, `profit_ratio` |
| Log transforms | `log_budget`, `log_revenue`, `log_runtime`, `log_popularity`, `log_vote_count` |

To avoid leakage or invalid model inputs, Agent 2 removes raw text, raw JSON-like columns, and the original `vote_average` column after creating the target.

Agent 2 also performs feature selection by:

1. removing highly correlated redundant features;
2. ranking numeric features by absolute correlation with the target;
3. keeping the top selected predictors requested by the LLM feature plan.

Outputs:

```text
outputs/engineered_data.csv
outputs/feature_report.txt
```

### Agent 3: Model Trainer — “The Coder”

The Model Trainer uses the LLM to generate executable Python code for XGBoost training.

Feedback loop:

1. Generate baseline XGBoost code.
2. Execute the generated code.
3. Read Accuracy, Recall, and F1.
4. Decide whether the result is good enough.
5. If not, generate new code with adjusted hyperparameters.
6. Stop when performance is acceptable or when the maximum number of attempts is reached.

Outputs:

```text
outputs/model_logs.txt
outputs/final_report.md
```

---

## Latest Model Results

The most recent run used three XGBoost attempts. Attempt 3 produced the best balance of recall and F1.

| Attempt | Accuracy | Recall | F1 Score | Decision |
|---:|---:|---:|---:|---|
| 1 | 0.9469 | 0.4853 | 0.5641 | Continue |
| 2 | 0.9542 | 0.5000 | 0.6071 | Continue |
| 3 | 0.9521 | 0.6029 | 0.6406 | Stop |

**Best result:** Attempt 3.

Attempt 3 has slightly lower accuracy than Attempt 2, but it has much better recall and the best F1 score. This matters because highly rated movies are the positive class, and recall measures how many truly highly rated movies the model successfully finds.

---

## Project Structure

```text
multi-agent-automl-movie-rating/
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
git clone https://github.com/Ketishavadze/multi-agent-automl-movie-rating.git
cd multi-agent-automl-movie-rating
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

Create a `.env` file in the project root:

```text
OPENAI_API_KEY=your_api_key_here
```

Do not commit `.env` to GitHub.

---

## Running the Project

Run the full pipeline:

```bash
python main.py
```

This runs all three agents sequentially:

1. Agent 1: Data Cleaner
2. Agent 2: Feature Engineer
3. Agent 3: Model Trainer

After execution, generated files appear in the `outputs/` folder.

---

## Assignment Checklist

- [x] Agent 1 cleans the data and passes `clean_data.csv` plus a structured cleaning report to Agent 2.
- [x] Agent 2 creates new movie-specific features using domain logic.
- [x] Agent 2 performs feature selection to reduce redundancy.
- [x] Agent 3 generates executable Python code.
- [x] Agent 3 executes the code and reads model metrics.
- [x] Agent 3 reacts to model results through a feedback loop.
- [x] The system saves execution logs and a final Markdown report.

---

## Important Notes

- The LLM does not directly predict movie ratings.
- The LLM agents make cleaning, feature-engineering, and modeling decisions.
- XGBoost performs the final classification task.
- The model predicts a binary label: highly rated or not highly rated.
- `vote_average` is removed before training to avoid direct target leakage.
- Recall and F1 are emphasized because the positive class is smaller than the negative class.
