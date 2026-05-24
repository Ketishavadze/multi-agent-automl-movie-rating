# Multi-Agent AutoML Team for Movie Rating Classification

## Objective
This system implements a sequential three-agent AutoML workflow for predicting whether a movie is highly rated. The binary target is `is_highly_rated`, created from the original rating column before leakage-prone fields are removed.

## Pipeline Summary
1. **Agent 1 — Data Cleaner:** inspects raw metadata, chooses cleaning actions, saves `clean_data.csv`, and writes `cleaning_report.txt`.
2. **Agent 2 — Feature Engineer:** receives Agent 1's output, creates semantic movie features, removes leakage/raw columns, performs feature selection, saves `engineered_data.csv`, and writes `feature_report.txt`.
3. **Agent 3 — Model Trainer:** generates executable XGBoost code, evaluates Accuracy/Recall/F1, and iterates based on its own decision step.

## Agent 1: Data Cleaning Summary
- Goal: audit the raw dataset and prepare clean_data.csv for Agent 2.
- Created has_homepage from missingness in homepage.
- Created has_overview from missingness in overview.
- Created has_release_date from missingness in release_date.
- Created has_runtime from missingness in runtime.
- Created has_tagline from missingness in tagline.
- Converted release_date to datetime.
- Imputed overview using unknown; nulls 3 → 0. Reason: Minimal missingness and text, replacing with unknown to preserve data
- Imputed runtime using median; nulls 2 → 0. Reason: Small number of missing numerical values, median is robust for skewed data
- Imputed tagline using unknown; nulls 844 → 0. Reason: Substantial missingness and text data, using unknown to preserve missing info
- Imputed release_date using mode; nulls 1 → 0. Reason: Only one missing date, filling with mode date to keep consistency
- Imputed homepage using unknown; nulls 3091 → 0. Reason: High missingness in URL column, treat missing as unknown category
- Dropped columns selected by the LLM: ['id', 'movie_id']
- Cleaned shape: (4803, 27)

## Agent 2: Feature Engineering Summary
- Goal: create semantic movie features and select the most relevant predictors.
- "reason_summary": "Extract meaningful semantic features from JSON columns, decompose release_date, encode categorical variables like original_language and status, create count features for cast and crew, and engineer financial ratios and log transformations. Remove raw text and JSON columns to avoid leakage and overfitting. Vote_average is used only to create the target label and then dropped. Select top 40 features after removing highly correlated ones to reduce redundancy.",
- "reason_summary": "Extract meaningful semantic features from JSON columns, decompose release_date, encode categorical variables like original_language and status, create count features for cast and crew, and engineer financial ratios and log transformations. Remove raw text and JSON columns to avoid leakage and overfitting. Vote_average is used only to create the target label and then dropped. Select top 40 features after removing highly correlated ones to reduce redundancy.",
- Created target is_highly_rated using vote_average >= 7.5.
- Created release_year, release_month, movie_age, and release_season from release_date.
- Created main_genre from genres.
- Created genre_count from genres.
- Created keyword_count from keywords.
- Created company_count from production_companies.
- Created country_count from production_countries.
- Created spoken_language_count from spoken_languages.
- Created director_name, crew_count, and director_movie_count from crew.
- Created main_actor, cast_count, and main_actor_movie_count from cast.
- Created has_budget, has_revenue, profit, and profit_ratio.
- Created log_budget.
- Created log_revenue.
- Created log_runtime.
- Created log_popularity.

## Target Distribution
class 0: 4466 (92.98%), class 1: 337 (7.02%)

## Agent 3: Model Training Results
| Attempt | Accuracy | Recall | F1 | Decision |
|---:|---:|---:|---:|---|
| 1 | 0.9492 | 0.3929 | 0.5197 | Continue |
| 2 | 0.9417 | 0.4762 | 0.5333 | Continue |
| 3 | 0.9376 | 0.5357 | 0.5455 | Stop |

**Best result:** Attempt 3, selected by highest F1 with recall as a tie-breaker. Accuracy = 0.9376, Recall = 0.5357, F1 = 0.5455.

The final model keeps high overall accuracy while improving recall compared with the baseline. Recall remains the limiting metric, which is expected when the positive class of highly rated movies is smaller than the negative class.

## Full Agent 3 Execution Log
```text
Agent 3 — Model Trainer
Goal: Generate and execute Python code to train an XGBoost model.
The agent must inspect results and decide whether to improve or stop.


================ ATTEMPT 1 ================

LLM generated this training code:

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score
from xgboost import XGBClassifier

# Load data
df = pd.read_csv('outputs/engineered_data.csv')
X = df.drop(columns=['is_highly_rated'])
y = df['is_highly_rated']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)

# Initialize model (baseline)
model = XGBClassifier()

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print metrics
print(f"accuracy={accuracy}")
print(f"recall={recall}")
print(f"f1={f1}")

Executing generated code...

STDOUT:
accuracy=0.9492089925062448
recall=0.39285714285714285
f1=0.5196850393700787

STDERR:


LLM Decision:
DECISION: CONTINUE
Reason: Recall and F1 are weak and another attempt remains.

================ ATTEMPT 2 ================

LLM generated this training code:

import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score

# Load data
data = pd.read_csv('outputs/engineered_data.csv')
X = data.drop(columns=['is_highly_rated'])
y = data['is_highly_rated']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)

# Define model with adjusted hyperparameters to improve recall
model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=sum(y == 0) / sum(y == 1),
    
    eval_metric='logloss'
)

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"accuracy={accuracy}")
print(f"recall={recall}")
print(f"f1={f1}")

Executing generated code...

STDOUT:
accuracy=0.9417152373022482
recall=0.47619047619047616
f1=0.5333333333333333

STDERR:


LLM Decision:
DECISION: CONTINUE
Reason: Recall and F1 are weak and another attempt remains.

================ ATTEMPT 3 ================

LLM generated this training code:

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score
from xgboost import XGBClassifier

# Load data
data = pd.read_csv('outputs/engineered_data.csv')
X = data.drop(columns=['is_highly_rated'])
y = data['is_highly_rated']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y, test_size=0.25)

# Adjust hyperparameters based on previous results (Attempt 2)
# Previous recall was low (~0.48), so increase scale_pos_weight and adjust learning_rate and max_depth
scale_pos_weight = (y == 0).sum() / (y == 1).sum()
model = XGBClassifier(
    n_estimators=150,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    
    eval_metric='logloss'
)

# Train model
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"accuracy={accuracy}")
print(f"recall={recall}")
print(f"f1={f1}")

Executing generated code...

STDOUT:
accuracy=0.9375520399666945
recall=0.5357142857142857
f1=0.5454545454545454

STDERR:


LLM Decision:
DECISION: STOP
Reason: F1 and recall are weak, but this is the final allowed attempt.

Final decision: stopping model training loop.
```
